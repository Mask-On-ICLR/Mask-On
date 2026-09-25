"""Frozen single-delta baselines and Mask-On materialization."""

from pathlib import Path
import math
import shutil

import torch
from safetensors.torch import load_file, save_file

from .artifacts import (
    atomic_json,
    content_sha,
    finish,
    read_json,
    run_directory,
    runtime,
    verify_done,
)
from .checkpoints import Checkpoint, identity, materialize
from .download import catalog, snapshot
from .merge.linear import RestrictedTensorSource
from .merge.support import signed_top_support

METHODS = (
    "task_arithmetic",
    "ties",
    "dare",
    "t_switch",
    "mask_on",
    "adamerging",
    "adamerging_plus_plus",
    "bitdelta",
    "delta_come",
    "regmean",
    "aim",
)


def is_vocab(key):
    return any(x in key for x in ("embed_tokens", "word_embeddings", "lm_head", "output_layer"))


class Candidate:
    def __init__(
        self,
        base,
        endpoint,
        method,
        *,
        beta=0.4,
        alpha=0.5,
        state=None,
        seed=20260909,
        drop=0.9,
        vocabulary="endpoint",
        payload_root=None,
    ):
        if method not in METHODS or not 0 <= beta <= 1 or not 0 <= alpha < 1 or not 0 <= drop < 1:
            raise ValueError("Invalid method/coefficient")
        self.base, self.endpoint, self.keys = base, endpoint, endpoint.keys
        self.method, self.beta, self.alpha, self.drop = method, beta, alpha, drop
        self.vocabulary, self.state = vocabulary, state or {}
        self.rng = torch.Generator().manual_seed(seed)
        self.threshold = None
        aligned = frozenset(
            k for k in base.keys & endpoint.keys if base.shape(k) == endpoint.shape(k)
        )
        self.aligned = aligned
        if method in ("ties", "adamerging_plus_plus"):
            from .merge.ties import GlobalTiesThresholdAuditor

            a, d = RestrictedTensorSource(base, aligned), RestrictedTensorSource(endpoint, aligned)
            receipt = GlobalTiesThresholdAuditor(
                retention_fractions=[0.2], chunk_elements=1024**2
            ).audit(a, a, d)[0.2]
            self.threshold = receipt.diffusion_threshold
        self._dare_order = iter(sorted(self.keys))
        self.payload_root, self.payload_index = payload_root, {}

    def _record_payload(self, key, target, code=None, scale=None):
        if self.payload_root is None or self.method not in ("mask_on", "t_switch"):
            return
        from .merge.delta_compression import pack_codes

        root = Path(self.payload_root)
        root.mkdir(parents=True, exist_ok=True)
        name = f"{len(self.payload_index):05d}.safetensors"
        if code is None:
            tensors = {"exact": target.contiguous()}
            kind = "exact_native_exception"
        else:
            tensors = {
                "code": pack_codes((code.reshape(-1) + 1).to(torch.uint8), 2),
                "scale": torch.as_tensor(scale).float().cpu().contiguous(),
            }
            kind = "ternary_delta_2bit"
        save_file(tensors, str(root / name))
        self.payload_index[key] = dict(
            file=name, kind=kind, shape=list(target.shape), dtype=str(target.dtype)
        )

    def tensor(self, key):
        target = self.endpoint.tensor(key)
        if key not in self.aligned or not target.is_floating_point():
            self._record_payload(key, target)
            return target
        if self.vocabulary == "endpoint" and is_vocab(key):
            self._record_payload(key, target)
            return target
        base = self.base.tensor(key).float()
        delta = target.float() - base
        method = self.method
        if method in (
            "task_arithmetic",
            "ties",
            "dare",
            "adamerging",
            "adamerging_plus_plus",
            "aim",
        ):
            update = delta
            if self.threshold is not None:
                update = update * (update.abs() >= self.threshold)
            if method == "dare":
                keep = 1 - torch.bernoulli(torch.full_like(delta, self.drop), generator=self.rng)
                update = keep * delta / (1 - self.drop)
            coef = self.beta
            if method.startswith("adamerging"):
                coef = self.state["coefficients"][key]
                if not math.isfinite(coef) or not 0 <= coef <= 1:
                    raise ValueError("Invalid fitted coefficient")
            if method == "aim" and not is_vocab(key):
                importance = self.state["importance"][key]
                if (
                    importance.shape != delta.shape[-1:]
                    or not torch.isfinite(importance).all()
                    or (importance < 0).any()
                    or (importance > 1).any()
                ):
                    raise ValueError("Missing/invalid AIM input importance " + key)
                update = update * (1 - importance * 0.6)
            return (base + coef * update).to(target.dtype)
        if method in ("bitdelta", "delta_come"):
            return self.state["compressed"].tensor(key).to(target.dtype)
        if method == "regmean":
            if delta.ndim != 2 or is_vocab(key):
                return ((base + target.float()) / 2).to(target.dtype)
            from .merge.regmean_full import merge_from_full_grams

            row = self.state["grams"][key]  # Load the Gram entry for this parameter.
            return merge_from_full_grams(
                base,
                target.float(),
                row["ar"],
                row["diffusion"],
                relative_loading=self.state.get("relative_loading", 0.0),
            ).to(target.dtype)
        flat = delta.reshape(delta.shape[0], -1) if delta.ndim > 1 else delta.reshape(1, -1)
        active = signed_top_support(flat, self.alpha)
        code = active * flat.sign()
        if method == "t_switch":
            scale = torch.sqrt((flat * active).square().sum() / active.sum().clamp_min(1))
        elif key in self.state["scales"]:
            scale = self.state["scales"][key].reshape(-1, 1)
            if (
                scale.shape != (flat.shape[0], 1)
                or not torch.isfinite(scale).all()
                or (scale < 0).any()
            ):
                raise ValueError("Invalid nonnegative output-channel scale: " + key)
        elif delta.ndim != 2:
            scale = (flat * code).sum() / code.square().sum().clamp_min(1)
        else:
            raise ValueError("Missing calibrated projection; no silent weight-LS fallback: " + key)
        self._record_payload(key, target, code, scale)
        return (base + (code * scale).reshape_as(delta)).to(target.dtype)


def prepare(
    model, method, output, *, beta=None, alpha=0.5, state=None, drop=0.5, seed=0, vocabulary=None
):
    # Use endpoint vocabulary for representation methods and merged vocabulary otherwise.
    beta = (
        beta
        if beta is not None
        else (1.0 if method == "ties" else 0.5 if method == "dare" else 0.4)
    )
    vocabulary = vocabulary or (
        "endpoint" if method in ("mask_on", "t_switch", "bitdelta", "delta_come") else "full"
    )
    base_path, endpoint_path = snapshot(model, "ar"), snapshot(model, "diffusion")
    base_files, endpoint_files = identity(base_path), identity(endpoint_path)
    state_data, state_ref = {}, None
    if state:
        state = Path(state)
        verify_done(state)
        state_manifest = read_json(state / "manifest.json")
        if (
            state_manifest.get("model") != model
            or state_manifest.get("checkpoints") != catalog()["models"][model]
        ):
            raise ValueError("Fitted state model/anchor binding mismatch")
        if (
            state_manifest.get("base_files") != base_files
            or state_manifest.get("endpoint_files") != endpoint_files
        ):
            raise ValueError("Fitted state actual checkpoint file hashes mismatch")
        if method != "mask_on" and state_manifest.get("method") != method:
            raise ValueError("Fitted state baseline mismatch")
        state_ref = content_sha(read_json(state / "DONE.json"))
        if method == "mask_on":
            alpha = read_json(state / "selection.json")["alpha"]
            state_data["scales"] = load_file(str(state / "scales.safetensors"))
        elif method.startswith("adamerging"):
            state_data["coefficients"] = read_json(state / "coefficients.json")
            if state_manifest.get("selection") != "minimum_validation_loss_earliest_tie":
                raise ValueError("Ada coefficients must be validation-selected")
        elif method == "aim":
            state_data["importance"] = load_file(str(state / "importance.safetensors"))
        elif method == "regmean":
            values = load_file(str(state / "grams.safetensors"))
            keys = {k.split("::", 1)[1] for k in values}
            state_data["grams"] = {
                k: {mode: values[mode + "::" + k] for mode in ("ar", "diffusion")} for k in keys
            }
            state_data["relative_loading"] = state_manifest.get("relative_loading", 0.0)
        elif method in ("bitdelta", "delta_come"):
            state_data["compressed_root"] = state / "compressed"
            state_data["compressed_receipt"] = read_json(state / "compressed/receipt.json")
            binding = state_data["compressed_receipt"]["binding"]
            if any(
                binding.get(k) != state_manifest.get(k) for k in ("model", "method", "checkpoints")
            ):
                raise ValueError("Compressed inner/outer checkpoint binding mismatch")
        else:
            raise ValueError(
                "Use the fitted-state API described in README.md for " + method
            )
    if (
        method
        in (
            "mask_on",
            "adamerging",
            "adamerging_plus_plus",
            "aim",
            "regmean",
            "bitdelta",
            "delta_come",
        )
        and not state_data
    ):
        raise ValueError(
            method + " requires fitted state; see README.md (never uses guessed parameters)"
        )
    contract = dict(
        model=model,
        method=method,
        checkpoints=catalog()["models"][model],
        base_files=base_files,
        endpoint_files=endpoint_files,
        beta=beta,
        alpha=alpha,
        drop=drop,
        seed=seed,
        vocabulary=vocabulary,
        state_sha256=state_ref,
        runtime=runtime(),
    )
    with run_directory(output, contract) as root:
        if (root / "DONE.json").exists():
            return verify_done(root)
        needed = sum(p.stat().st_size for p in endpoint_path.glob("*.safetensors"))
        if method in ("mask_on", "t_switch"):
            needed = int(needed * 1.5)  # Count compact codes and endpoint vocabulary tensors.
        if shutil.disk_usage(root).free < needed + 2 * 1024**3:
            raise OSError("Insufficient space for one dense preparation plus 2 GiB margin")
        if (root / "checkpoint").exists():
            raise ValueError(
                "Interrupted checkpoint writer: preserve attempt and use a fresh output root"
            )
        with Checkpoint(base_path) as base, Checkpoint(endpoint_path) as endpoint:
            if "compressed_root" in state_data:
                from .merge.delta_compression import CompressedDeltaSource
                from .artifacts import file_sha

                compressed = state_data["compressed_root"]
                state_data["compressed"] = CompressedDeltaSource(
                    base,
                    endpoint,
                    compressed,
                    receipt_sha256=file_sha(compressed / "receipt.json"),
                    expected_binding=state_data["compressed_receipt"]["binding"],
                )
            candidate = Candidate(
                base,
                endpoint,
                method,
                beta=beta,
                alpha=alpha,
                state=state_data,
                seed=seed,
                drop=drop,
                vocabulary=vocabulary,
                payload_root=root / "payload",
            )
            files = materialize(candidate, endpoint_path, root / "checkpoint")
            payload_files = []
            if candidate.payload_index:
                atomic_json(
                    root / "payload/index.json",
                    dict(
                        schema=1,
                        entries=candidate.payload_index,
                        anchor=contract["checkpoints"]["anchor"],
                        endpoint=contract["checkpoints"]["endpoint"],
                        anchor_files=contract["base_files"],
                        alpha=alpha,
                        packing="dense_two_bit_codes_minus_one_zero_plus_one",
                        bitwise_reconstruction=True,
                    ),
                )
                payload_files = ["payload/" + p.name for p in sorted((root / "payload").iterdir())]
        atomic_json(root / "receipt.json", dict(contract_sha256=content_sha(contract), files=files))
        finish(
            root,
            ["manifest.json", "receipt.json"] + ["checkpoint/" + k for k in files] + payload_files,
        )
        return read_json(root / "receipt.json")
