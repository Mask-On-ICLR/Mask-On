"""Fit, serialize, and reconstruct per-parameter AdaMerging coefficients.

Shared tensors use the diffusion-minus-AR delta; native-only tensors retain
endpoint values.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.nn.utils import parametrize


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class LayerwiseAdaConfig:
    variant: str = "adamerging"
    initial_coefficient: float = 0.3
    learning_rate: float = 1e-3
    steps: int = 500
    validation_interval: int = 25
    seed: int = 20260909
    # Enable sparse FP32 subtraction residuals when configured.
    compensate_endpoint_cancellation: bool = False

    def validate(self):
        if self.variant not in ("adamerging", "adamerging_plus_plus"):
            raise ValueError("unregistered Ada variant")
        if not 0 <= self.initial_coefficient <= 1:
            raise ValueError("initial coefficient outside [0,1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("invalid learning rate")
        if self.steps < 1 or self.validation_interval < 1:
            raise ValueError("invalid optimizer/validation budget")


class LayerwiseCoefficients(torch.nn.Module):
    def __init__(self, keys, config, *, device="cpu"):
        super().__init__()
        config.validate()
        self.keys = tuple(keys)
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("empty/duplicate coefficient mapping")
        self.config = config
        self.raw = torch.nn.Parameter(
            torch.full(
                (len(self.keys),), config.initial_coefficient, device=device, dtype=torch.float32
            )
        )

    def forward(self):
        return self.raw.clamp(0, 1)

    @torch.no_grad()
    def project(self):
        self.raw.clamp_(0, 1)


def endpoint_cancellation_correction(base, delta, endpoint):
    """Return sparse residuals from FP32 endpoint-minus-base subtraction."""
    indices, values = [], []
    for start in range(0, base.numel(), 1024**2):
        end = start + 1024**2
        b = base.reshape(-1)[start:end].float()
        d = delta.reshape(-1)[start:end].to(base.device)
        e = endpoint.detach().reshape(-1)[start:end].float()
        residual = e - (b + d)
        selected = residual.ne(0).nonzero().flatten()
        indices.append(selected + start)
        values.append(residual[selected])
    return torch.cat(indices), torch.cat(values)


class DeltaCoefficient(torch.nn.Module):
    """Compute scaled deltas with gradients restricted to the coefficients."""

    def __init__(self, bank, index, delta, correction=None):
        super().__init__()
        object.__setattr__(self, "bank", bank)
        self.index = index
        self.register_buffer("delta", delta.detach())
        self.register_buffer("correction_indices", None if correction is None else correction[0])
        self.register_buffer("correction_values", None if correction is None else correction[1])

    def forward(self, base):
        return _CoefficientWeight.apply(
            base, self.bank()[self.index], self.delta,
            self.correction_indices, self.correction_values,
        )


class _CoefficientWeight(torch.autograd.Function):
    """Accumulate coefficient gradients in chunks with optional CPU delta storage."""

    @staticmethod
    def forward(ctx, base, coefficient, delta, correction_indices=None, correction_values=None):
        ctx.save_for_backward(delta, correction_indices, correction_values)
        ctx.input_count = len(ctx.needs_input_grad)
        ctx.coefficient_device = coefficient.device
        result = torch.empty(base.shape, device=base.device, dtype=base.dtype)
        for start in range(0, base.numel(), 1024**2):
            end = start + 1024**2
            b = base.reshape(-1)[start:end].float()
            d = delta.reshape(-1)[start:end].to(base.device)
            result.reshape(-1)[start:end].copy_(b + coefficient * d)
        if correction_indices is not None and correction_indices.numel():
            src_indices = correction_indices.to(delta.device)
            dst_indices = correction_indices.to(base.device)
            d = delta.reshape(-1)[src_indices].to(base.device)
            # Mask residual corrections at coordinates removed by TIES pruning.
            r = correction_values.to(base.device) * d.ne(0)
            corrected = (base.reshape(-1)[dst_indices].float() + coefficient * d) + coefficient * r
            result.reshape(-1)[dst_indices] = corrected.to(base.dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        delta, correction_indices, correction_values = ctx.saved_tensors
        result = torch.zeros((), device=ctx.coefficient_device, dtype=torch.float32)
        for start in range(0, delta.numel(), 1024**2):
            end = start + 1024**2
            result += (
                grad_output.reshape(-1)[start:end].float()
                * delta.reshape(-1)[start:end].to(grad_output.device)
            ).sum()
        if correction_indices is not None and correction_indices.numel():
            active = delta.reshape(-1)[correction_indices.to(delta.device)].ne(0)
            residual = correction_values.to(grad_output.device) * active.to(grad_output.device)
            result += (grad_output.reshape(-1)[correction_indices.to(grad_output.device)].float()
                       * residual).sum()
        return (None, result, None, None, None)[:ctx.input_count]


def install_layerwise(
    model,
    base_source,
    *,
    config,
    source_key=lambda key: key,
    ties_threshold=None,
    delta_device=None,
):
    """Install per-parameter coefficients with tied aliases and native keys.

    AdaMerging++ applies a global TIES threshold to the task vector.
    """
    config.validate()
    if (config.variant == "adamerging_plus_plus") != (ties_threshold is not None):
        raise ValueError("++ alone requires the frozen global TIES threshold")
    if ties_threshold is not None and (not math.isfinite(ties_threshold) or ties_threshold < 0):
        raise ValueError("invalid global TIES threshold")
    aliases = {}
    for key, value in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(value), []).append(key)
        value.requires_grad_(False)
    selected = []
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for names in aliases.values():
        aligned = [key for key in names if source_key(key) in base_source.keys]
        if not aligned:
            continue
        key = aligned[0]
        if parameters[key].is_floating_point():
            selected.append((key, names))
    # Validate parameter shapes and aliases before installing coefficients.
    for key, names in selected:
        base = base_source.tensor(source_key(key))
        if base.shape != parameters[key].shape or not torch.isfinite(base).all():
            raise ValueError("invalid aligned parameter " + key)
        for alias in names:
            alias_key = source_key(alias)
            if alias_key in base_source.keys and alias_key != source_key(key):
                if not torch.equal(base, base_source.tensor(alias_key)):
                    raise ValueError("inconsistent tied anchor aliases " + key)
    bank = LayerwiseCoefficients(
        [key for key, _ in selected], config, device=next(model.parameters()).device
    )
    handles, mapping = [], {}
    for i, (key, names) in enumerate(selected):
        weight = parameters[key]
        base = base_source.tensor(source_key(key))
        if base.shape != weight.shape:
            raise ValueError("unaligned parameter " + key)
        # Compute deltas and scaled weights in FP32.
        base = base.to(weight.device, dtype=weight.dtype)
        delta = weight.detach().float() - base.float()
        if not torch.isfinite(delta).all():
            raise ValueError("nonfinite endpoint delta " + key)
        correction = None
        if config.compensate_endpoint_cancellation:
            correction = endpoint_cancellation_correction(base, delta, weight)
        if ties_threshold is not None:
            delta.masked_fill_(delta.abs() < ties_threshold, 0)
        if delta_device is not None:
            delta = delta.to(delta_device)
        modulation = DeltaCoefficient(bank, i, delta, correction)
        with torch.no_grad():
            weight.copy_(base)
        for name in names:
            parent, _, attr = name.rpartition(".")
            module = model.get_submodule(parent) if parent else model
            parametrize.register_parametrization(module, attr, modulation)
            handles.append((module, attr))
        mapping[key] = dict(source_key=source_key(key), aliases=names, shape=list(weight.shape))
    return bank, mapping, handles


class BestValidation:
    """Track minimum validation loss and retain the earliest state on ties."""

    def __init__(self):
        self.loss, self.step, self.values = math.inf, None, None

    def consider(self, *, step, loss, bank):
        if not math.isfinite(loss):
            raise ValueError("nonfinite validation loss")
        values = bank().detach().cpu().clone()
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite coefficients")
        if loss < self.loss:
            self.loss, self.step, self.values = float(loss), int(step), values
            return True
        return False

    def save(self, root, *, bank, mapping, binding, trace):
        if self.values is None:
            raise ValueError("no validation-selected state")
        required = {
            "model",
            "endpoint_revision",
            "anchor_revision",
            "panel_sha256",
            "objective_sha256",
            "source_worktree_sha256",
            "baseline_recipe_sha256",
        }
        if not required <= binding.keys() or any(not binding[k] for k in required):
            raise ValueError("incomplete reusable coefficient identity")
        if set(mapping) != set(bank.keys):
            raise ValueError("coefficient mapping mismatch")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        save_file({"coefficients": self.values}, str(root / "coefficients.safetensors"))
        receipt = dict(
            schema_version=1,
            variant=bank.config.variant,
            config=asdict(bank.config),
            binding=binding,
            keys=list(bank.keys),
            mapping=mapping,
            best_step=self.step,
            validation_loss=self.loss,
            selection="minimum_validation_loss_earliest_tie",
            coefficient_sha256=file_digest(root / "coefficients.safetensors"),
            preparation_key=digest(
                dict(binding=binding, config=asdict(bank.config), mapping=mapping)
            ),
            trace=trace,
            dense_weights_saved=False,
            refit_on_evaluation=False,
        )
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (root / "DONE").write_text("DONE\n")
        return receipt


def load_coefficients(root, *, expected_receipt_sha256, expected_binding):
    root = Path(root)
    if not (root / "DONE").is_file() or any((root / n).exists() for n in ("FAILED", "FAILED.json")):
        raise ValueError("coefficient bundle is not terminal")
    if file_digest(root / "receipt.json") != expected_receipt_sha256:
        raise ValueError("coefficient receipt drift")
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt["binding"] != expected_binding:
        raise ValueError("coefficient endpoint/data/objective binding mismatch")
    if file_digest(root / "coefficients.safetensors") != receipt["coefficient_sha256"]:
        raise ValueError("coefficient payload drift")
    tensors = load_file(str(root / "coefficients.safetensors"))
    if set(tensors) != {"coefficients"}:
        raise ValueError("unexpected payload tensors")
    values = tensors["coefficients"]
    if (
        values.shape != (len(receipt["keys"]),)
        or len(set(receipt["keys"])) != len(values)
        or not torch.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("invalid coefficient values/mapping")
    if set(receipt["mapping"]) != set(receipt["keys"]):
        raise ValueError("coefficient mapping mismatch")
    if not math.isfinite(receipt["validation_loss"]):
        raise ValueError("invalid selected validation loss")
    return dict(zip(receipt["keys"], values.tolist(), strict=True)), receipt


def prepare_ties_basis(base_source, endpoint_source, mapping):
    """Compute the global TIES threshold over unique aligned parameter coordinates."""
    from mask_on.merge.ties import GlobalTiesThresholdAuditor

    class View:
        keys = frozenset(mapping)

        def __init__(self, source, anchor):
            self.source, self.anchor = source, anchor

        def tensor(self, key):
            if self.anchor:
                return self.source.tensor(mapping[key]["source_key"])
            stored = [name for name in mapping[key]["aliases"] if name in self.source.keys]
            if not stored:
                raise ValueError("no checkpoint source for tied parameter " + key)
            value = self.source.tensor(stored[0])
            for alias in stored[1:]:
                if not torch.equal(value, self.source.tensor(alias)):
                    raise ValueError("inconsistent tied endpoint aliases " + key)
            return value

    base, endpoint = View(base_source, True), View(endpoint_source, False)
    audit = GlobalTiesThresholdAuditor(retention_fractions=[0.2], chunk_elements=1024**2)
    result = asdict(audit.audit(base, base, endpoint)[0.2])
    return dict(
        result,
        mapping_sha256=digest(mapping),
        specialization="single_nonzero_delta_global_trim_ties_retained",
    )


class SelectedAdaTensorSource:
    """Reconstruct tensors on CPU from saved coefficients and endpoint weights."""

    def __init__(
        self, base_source, endpoint_source, root, *, expected_receipt_sha256, expected_binding
    ):
        self.values, self.receipt = load_coefficients(
            root, expected_receipt_sha256=expected_receipt_sha256, expected_binding=expected_binding
        )
        self.base, self.endpoint = base_source, endpoint_source
        self.aliases = {}
        for key, item in self.receipt["mapping"].items():
            if item["source_key"] not in self.base.keys or not any(
                a in self.endpoint.keys for a in item["aliases"]
            ):
                raise ValueError("selected coefficient source key missing")
            for alias in item["aliases"]:
                if alias in self.aliases:
                    raise ValueError("overlapping coefficient aliases")
                self.aliases[alias] = key
        pp = self.receipt["variant"] == "adamerging_plus_plus"
        basis = self.receipt["binding"].get("ties_basis")
        if pp and (
            not basis
            or basis.get("retention_fraction") != 0.2
            or basis.get("mapping_sha256") != digest(self.receipt["mapping"])
        ):
            raise ValueError("missing or inconsistent frozen global TIES basis")
        self.threshold = basis["diffusion_threshold"] if pp else None
        if self.threshold is not None and (not math.isfinite(self.threshold) or self.threshold < 0):
            raise ValueError("invalid frozen TIES threshold")

    @property
    def keys(self):
        return self.endpoint.keys

    def tensor(self, key):
        endpoint = self.endpoint.tensor(key)
        if key not in self.aliases:
            return endpoint
        canonical = self.aliases[key]
        item = self.receipt["mapping"][canonical]
        base = self.base.tensor(item["source_key"]).to(dtype=endpoint.dtype)
        if list(endpoint.shape) != item["shape"] or endpoint.shape != base.shape:
            raise ValueError("selected coefficient tensor shape mismatch")
        delta = endpoint.float() - base.float()
        correction = None
        if self.receipt["config"].get("compensate_endpoint_cancellation", False):
            correction = endpoint_cancellation_correction(base, delta, endpoint)
        if self.threshold is not None:
            delta.masked_fill_(delta.abs() < self.threshold, 0)
        result = base.float() + self.values[canonical] * delta
        if correction is not None:
            indices, values = correction
            result.reshape(-1)[indices] += (self.values[canonical] * values
                                           * delta.reshape(-1)[indices].ne(0))
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite selected coefficient reconstruction")
        return result.to(endpoint.dtype)


def verify_evaluation_coefficient_reference(
    candidate, *, model, baseline, recipe_sha256, endpoint_revision, namespace_alias=None
):
    """Validate a candidate's coefficient hashes, model identity, and selection rule."""
    ref = candidate.get("coefficient_bundle")
    if (
        not isinstance(ref, dict)
        or not {"root", "receipt_sha256", "binding", "fit_root", "fit_aggregate_sha256"}
        <= ref.keys()
    ):
        raise ValueError("Ada evaluation requires a frozen coefficient bundle reference")
    from mask_on.evaluation.adamerging_fitting import audit_completed_fit

    audit = audit_completed_fit(
        ref["fit_root"], expected_aggregate_sha256=ref["fit_aggregate_sha256"]
    )
    if audit["selected_receipt_sha256"] != ref["receipt_sha256"]:
        raise ValueError("Ada candidate does not reference the completed fit winner")
    _, receipt = load_coefficients(
        ref["root"], expected_receipt_sha256=ref["receipt_sha256"], expected_binding=ref["binding"]
    )
    binding = receipt["binding"]
    accepted_models = {model}
    if (
        isinstance(namespace_alias, Mapping)
        and namespace_alias.get("accepted")
        and namespace_alias.get("model") == model
        and namespace_alias.get("binding_model")
    ):
        accepted_models.add(namespace_alias["binding_model"])
    if (
        receipt["variant"] != baseline
        or binding["model"] not in accepted_models
        or binding["endpoint_revision"] != endpoint_revision
        or binding["baseline_recipe_sha256"] != recipe_sha256
    ):
        raise ValueError("Ada evaluation coefficient identity mismatch")
    if (
        receipt["selection"] != "minimum_validation_loss_earliest_tie"
        or receipt["refit_on_evaluation"] is not False
    ):
        raise ValueError(
            "Ada evaluation must reuse validation-selected coefficients without refitting"
        )
    return receipt
