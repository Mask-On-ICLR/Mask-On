"""Fit mode-specific low-rank adapters for decoder representations.

Apply z - Phi(z) after the final normalization and before the output head
with the carrier model's weights frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from mask_on.merge.linear import MergeError

SurgeryMode = Literal["ar", "diffusion"]
_MODES: tuple[SurgeryMode, ...] = ("ar", "diffusion")


@dataclass(frozen=True, slots=True)
class RepresentationSurgeryConfig:
    """Optimizer, adapter, and batch settings for representation surgery.

    The defaults use 500 iterations with two minibatches per mode per iteration.
    """

    hidden_size: int = 3584
    rank: int = 16
    iterations: int = 500
    batches_per_iteration: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    seed: int = 20260806

    def validate(self) -> None:
        if self.hidden_size < 1 or not 1 <= self.rank <= self.hidden_size:
            raise MergeError("representation-surgery dimensions are invalid")
        if self.iterations < 1 or self.batches_per_iteration < 1 or self.batch_size < 1:
            raise MergeError(
                "representation-surgery iterations, batches, and batch size must be positive"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise MergeError("representation-surgery learning rate must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise MergeError("representation-surgery Adam betas must lie in [0, 1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise MergeError("representation-surgery weight decay must be non-negative")


@dataclass(frozen=True, slots=True)
class RepresentationFeatureSet:
    """Detached final-layer feature rows and their example indices."""

    values: torch.Tensor
    example_indices: torch.Tensor
    row_identity_sha256: str

    def validate(self, *, hidden_size: int) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != hidden_size:
            raise MergeError("representation features have the wrong hidden dimension")
        if self.example_indices.ndim != 1 or self.example_indices.shape[0] != self.values.shape[0]:
            raise MergeError("representation feature/example rows do not align")
        if self.values.shape[0] < 2 or not bool(torch.isfinite(self.values).all()):
            raise MergeError("representation features must contain finite paired rows")
        if self.example_indices.dtype != torch.int64:
            raise MergeError("representation example indices must be int64")
        if len(self.row_identity_sha256) != 64:
            raise MergeError("representation row identity must be SHA-256")


class RepresentationSurgeryAdapter(nn.Module):
    """Two-layer bottleneck adapter that subtracts its predicted representation offset."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        if hidden_size < 1 or not 1 <= rank <= hidden_size:
            raise ValueError("invalid representation-surgery adapter dimensions")
        self.hidden_size = hidden_size
        self.rank = rank
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def predicted_bias(self, representation: torch.Tensor) -> torch.Tensor:
        return self.up_proj(F.relu(self.down_proj(representation)))

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.shape[-1] != self.hidden_size:
            raise MergeError("representation-surgery input has the wrong hidden size")
        return representation - self.predicted_bias(representation)


@dataclass(frozen=True, slots=True)
class SurgeryModeFit:
    """Fitted adapter and held-out diagnostics for one runtime mode."""

    mode: SurgeryMode
    train_rows: int
    validation_rows: int
    initial_train_l1: float
    final_train_l1: float
    initial_validation_l1: float
    final_validation_l1: float
    validation_gain: float
    loss_trace: tuple[float, ...]
    convergence_trace: tuple[dict[str, float | int | None], ...] = ()
    convergence_status: str = "NOT_EVALUATED"
    train_relative_reduction: float = 0.0
    validation_relative_reduction: float = 0.0
    tail_train_relative_range: float = 0.0


@dataclass(frozen=True, slots=True)
class RepresentationSurgeryFit:
    """CPU adapter states and diagnostics for AR and diffusion modes."""

    config: RepresentationSurgeryConfig
    state: dict[str, torch.Tensor]
    modes: dict[SurgeryMode, SurgeryModeFit]

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.state.values())


class RepresentationSurgeryFitter:
    """Fit adapter parameters from detached carrier and endpoint feature pairs."""

    def __init__(self, config: RepresentationSurgeryConfig) -> None:
        config.validate()
        self.config = config

    def fit(
        self,
        *,
        carrier: dict[SurgeryMode, RepresentationFeatureSet],
        expert: dict[SurgeryMode, RepresentationFeatureSet],
        validation_example_indices: dict[SurgeryMode, frozenset[int]],
        device: torch.device | str,
        progress_callback: Callable[[SurgeryMode, Mapping[str, Any]], None] | None = None,
    ) -> RepresentationSurgeryFit:
        if set(carrier) != set(_MODES) or set(expert) != set(_MODES):
            raise MergeError("representation surgery requires AR and diffusion features")
        state: dict[str, torch.Tensor] = {}
        results: dict[SurgeryMode, SurgeryModeFit] = {}
        for offset, mode in enumerate(_MODES):
            adapter, result = self._fit_mode(
                mode=mode,
                carrier=carrier[mode],
                expert=expert[mode],
                validation_examples=validation_example_indices[mode],
                device=torch.device(device),
                seed=self.config.seed + offset,
                progress_callback=progress_callback,
            )
            results[mode] = result
            for name, value in adapter.state_dict().items():
                state[f"{mode}.{name}"] = value.detach().float().cpu().contiguous()
        return RepresentationSurgeryFit(config=self.config, state=state, modes=results)

    def _fit_mode(
        self,
        *,
        mode: SurgeryMode,
        carrier: RepresentationFeatureSet,
        expert: RepresentationFeatureSet,
        validation_examples: frozenset[int],
        device: torch.device,
        seed: int,
        progress_callback: Callable[[SurgeryMode, Mapping[str, Any]], None] | None,
    ) -> tuple[RepresentationSurgeryAdapter, SurgeryModeFit]:
        carrier.validate(hidden_size=self.config.hidden_size)
        expert.validate(hidden_size=self.config.hidden_size)
        if carrier.row_identity_sha256 != expert.row_identity_sha256:
            raise MergeError(f"{mode} carrier/expert feature rows are not paired")
        if not torch.equal(carrier.example_indices, expert.example_indices):
            raise MergeError(f"{mode} carrier/expert example ownership differs")

        validation = torch.tensor(
            [int(value) in validation_examples for value in carrier.example_indices.tolist()],
            dtype=torch.bool,
        )
        train = ~validation
        if not bool(train.any()) or not bool(validation.any()):
            raise MergeError(f"{mode} fit requires non-empty disjoint train/validation rows")
        if set(carrier.example_indices[train].tolist()) & set(
            carrier.example_indices[validation].tolist()
        ):
            raise MergeError(f"{mode} examples leak across fit and validation")

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        adapter = RepresentationSurgeryAdapter(self.config.hidden_size, self.config.rank).to(
            device=device, dtype=torch.float32
        )
        optimizer = torch.optim.Adam(
            adapter.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )
        x_train = carrier.values[train].float().to(device)
        y_train = expert.values[train].float().to(device)
        x_validation = carrier.values[validation].float().to(device)
        y_validation = expert.values[validation].float().to(device)
        initial_train = float(F.l1_loss(x_train, y_train).item())
        initial_validation = float(F.l1_loss(x_validation, y_validation).item())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        trace: list[float] = []
        convergence_trace: list[dict[str, float | int | None]] = []

        def record_convergence(update_index: int, batch_l1: float | None) -> None:
            adapter.eval()
            with torch.inference_mode():
                train_l1 = float(F.l1_loss(adapter(x_train), y_train).item())
                validation_l1 = float(
                    F.l1_loss(adapter(x_validation), y_validation).item()
                )
            point: dict[str, float | int | None] = {
                "update": int(update_index),
                "batch_l1": batch_l1,
                "train_l1": train_l1,
                "validation_l1": validation_l1,
            }
            convergence_trace.append(point)
            if progress_callback is not None:
                progress_callback(mode, point)
            adapter.train()

        adapter.train()
        optimizer_updates = self.config.iterations * self.config.batches_per_iteration
        checkpoint_interval = max(1, optimizer_updates // 20)
        update = 0
        record_convergence(0, None)
        for _ in range(self.config.iterations):
            permutation = torch.randperm(x_train.shape[0], generator=generator)
            for batch_index in range(self.config.batches_per_iteration):
                start = batch_index * self.config.batch_size
                if start >= permutation.numel():
                    permutation = torch.randperm(x_train.shape[0], generator=generator)
                    start = 0
                indices = permutation[
                    start : min(start + self.config.batch_size, permutation.numel())
                ].to(device)
                loss = F.l1_loss(
                    adapter(x_train.index_select(0, indices)),
                    y_train.index_select(0, indices),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                update += 1
                batch_l1 = float(loss.detach().item())
                if update == 1 or update % checkpoint_interval == 0:
                    trace.append(batch_l1)
                if update % checkpoint_interval == 0 or update == optimizer_updates:
                    record_convergence(update, batch_l1)
        adapter.eval()
        with torch.inference_mode():
            final_train = float(F.l1_loss(adapter(x_train), y_train).item())
            final_validation = float(F.l1_loss(adapter(x_validation), y_validation).item())
        if not all(
            math.isfinite(value)
            for value in (initial_train, final_train, initial_validation, final_validation)
        ):
            raise MergeError(f"{mode} representation-surgery fit is non-finite")
        train_relative_reduction = (initial_train - final_train) / max(initial_train, 1e-12)
        validation_relative_reduction = (
            initial_validation - final_validation
        ) / max(initial_validation, 1e-12)
        tail_train = [float(point["train_l1"]) for point in convergence_trace[-5:]]
        tail_train_relative_range = (
            (max(tail_train) - min(tail_train)) / max(initial_train, 1e-12)
            if tail_train
            else math.inf
        )
        if final_train >= initial_train:
            convergence_status = "NOT_DECREASING"
        elif tail_train_relative_range <= 0.01:
            convergence_status = "DECREASING_AND_STABLE"
        else:
            convergence_status = "DECREASING_NOT_STABLE"
        result = SurgeryModeFit(
            mode=mode,
            train_rows=int(train.sum().item()),
            validation_rows=int(validation.sum().item()),
            initial_train_l1=initial_train,
            final_train_l1=final_train,
            initial_validation_l1=initial_validation,
            final_validation_l1=final_validation,
            validation_gain=initial_validation - final_validation,
            loss_trace=tuple(trace),
            convergence_trace=tuple(convergence_trace),
            convergence_status=convergence_status,
            train_relative_reduction=train_relative_reduction,
            validation_relative_reduction=validation_relative_reduction,
            tail_train_relative_range=tail_train_relative_range,
        )
        return adapter.cpu(), result


class ModeSpecificRepresentationSurgery:
    """Attach the selected mode's adapter to the final normalization output."""

    def __init__(self, adapters: dict[SurgeryMode, RepresentationSurgeryAdapter]) -> None:
        if set(adapters) != set(_MODES):
            raise ValueError("mode-specific surgery needs exactly AR and diffusion adapters")
        self.adapters = nn.ModuleDict(adapters)
        self._mode: SurgeryMode | None = None
        self._handle: Any | None = None

    def install(
        self,
        final_norm: nn.Module,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if self._handle is not None:
            raise MergeError("representation surgery is already installed")
        self.adapters.to(device=device, dtype=dtype)
        self._handle = final_norm.register_forward_hook(self._hook)

    def prepare_mode(self, mode: str) -> dict[str, Any]:
        if mode not in _MODES:
            raise ValueError(f"unsupported representation-surgery mode: {mode}")
        self._mode = mode  # type: ignore[assignment]
        return {"representation_surgery": True, "mode": mode, "mode_specific_state": True}

    def _hook(self, _: Any, __: Any, output: Any) -> Any:
        if self._mode is None:
            raise MergeError("representation-surgery mode was not selected before forward")
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise MergeError("final norm did not return a tensor")
        corrected = self.adapters[self._mode](value)
        return (corrected, *output[1:]) if isinstance(output, tuple) else corrected

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._mode = None


def save_representation_surgery_bundle(
    destination: str | Path,
    fit: RepresentationSurgeryFit,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Save adapter tensors and a JSON manifest with file hashes."""

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    tensor_path = destination / "adapter.safetensors"
    save_file(fit.state, str(tensor_path))
    tensor_sha256 = _sha256_file(tensor_path)
    receipt = {
        "schema_version": 1,
        "method": "Representation Surgery",
        "formal_definition": "z_hat_mode=z-Phi_mode(z); Phi=Up(ReLU(Down(z)))",
        "paper_default_rank": 16,
        "paper_appendix_iterations": 1000,
        "official_code_iterations": 500,
        "official_code_batches_per_task_per_iteration": 2,
        "optimizer_updates_per_mode": fit.config.iterations
        * fit.config.batches_per_iteration,
        "config": asdict(fit.config),
        "mode_results": {mode: asdict(result) for mode, result in fit.modes.items()},
        "adapter_parameter_count": fit.parameter_count,
        "mode_specific_adapter_count": 2,
        "adapter_sha256": tensor_sha256,
        "adapter_size_bytes": tensor_path.stat().st_size,
        "dense_checkpoint_saved": False,
        "strict_single_dense_state": False,
        **metadata,
    }
    receipt_path = destination / "bundle.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "bundle_sha256": _sha256_file(receipt_path)}


def load_representation_surgery_bundle(
    source: str | Path,
    *,
    expected_adapter_sha256: str | None = None,
) -> tuple[dict[SurgeryMode, RepresentationSurgeryAdapter], dict[str, Any]]:
    """Load adapter tensors and verify the bundle hashes."""

    source = Path(source).expanduser().resolve()
    receipt = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
    tensor_path = source / "adapter.safetensors"
    observed = _sha256_file(tensor_path)
    expected = expected_adapter_sha256 or str(receipt["adapter_sha256"])
    if observed != expected or str(receipt["adapter_sha256"]) != expected:
        raise MergeError("representation-surgery bundle tensor hash differs")
    config = RepresentationSurgeryConfig(**receipt["config"])
    config.validate()
    state = load_file(str(tensor_path), device="cpu")
    adapters: dict[SurgeryMode, RepresentationSurgeryAdapter] = {}
    for mode in _MODES:
        adapter = RepresentationSurgeryAdapter(config.hidden_size, config.rank)
        prefix = f"{mode}."
        mode_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        adapter.load_state_dict(mode_state, strict=True)
        adapter.eval()
        adapters[mode] = adapter
    return adapters, receipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
