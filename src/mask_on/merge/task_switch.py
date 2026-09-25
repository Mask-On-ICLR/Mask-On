"""Construct and reconstruct T-Switch task vectors (Qi et al., arXiv:2412.00054).

Filter positive and negative magnitudes per tensor, compute reconstruction
scales, and serialize the support and sign masks as grouped bit-packs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from mask_on.merge.linear import MergeError, TensorSource


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class TaskSwitchSerializedPayload:
    """Serialized task-vector metadata and payload files for one endpoint."""

    root: Path
    metadata: Mapping[str, Any]
    file_bytes: Mapping[str, int]
    file_sha256: Mapping[str, str]

    @property
    def additional_bytes(self) -> int:
        return sum(int(value) for value in self.file_bytes.values())


@dataclass(frozen=True, slots=True)
class TaskSwitchApplicationReceipt:
    """Reconstructed endpoint identity and tensor approximation errors."""

    tensor_sha256: str
    materialized_parameter_count: int
    endpoint_exact: bool | None
    max_abs_error: float | None
    mean_squared_error: float | None


@dataclass(frozen=True, slots=True)
class TaskSwitchTensorBits:
    """Support bits, sign bits, and scales for one parameter tensor.

    Expand the grouped payload into two little-endian bit-packs indexed by
    flat parameter position.
    """

    key: str
    shape: tuple[int, ...]
    activation_bitpack: torch.Tensor
    polarity_bitpack: torch.Tensor
    scale: float
    scale_values: torch.Tensor | None = None
    scale_granularity: str = "tensor"

    def broadcast_scale(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return a scalar or output-row scale reshaped for tensor broadcasting."""

        if self.scale_granularity == "tensor":
            return torch.tensor(self.scale, dtype=torch.float32, device=device)
        if self.scale_granularity != "row" or len(self.shape) < 2:
            raise MergeError(
                f"unsupported T-Switch scale granularity for {self.key}: "
                f"{self.scale_granularity}"
            )
        if self.scale_values is None or self.scale_values.numel() != self.shape[0]:
            raise MergeError(f"row-scale payload is incomplete for {self.key}")
        return self.scale_values.to(device=device, dtype=torch.float32).reshape(
            (self.shape[0],) + (1,) * (len(self.shape) - 1)
        )


class TaskSwitchBuilder:
    """Construct, serialize, and apply a task switch relative to a base checkpoint."""

    _GROUP_MARK_FILE = "group_mark.bitpack"
    _ACTIVATION_FILE = "activation.bitpack"
    _POLARITY_FILE = "polarity.bitpack"
    _SCALES_FILE = "scales.f32"
    _METADATA_FILE = "metadata.json"

    @classmethod
    def durable_payload_filenames(cls) -> frozenset[str]:
        """List mask, sign, scale, and metadata files in the serialized payload."""

        return frozenset(
            {
                cls._GROUP_MARK_FILE,
                cls._ACTIVATION_FILE,
                cls._POLARITY_FILE,
                cls._SCALES_FILE,
                cls._METADATA_FILE,
            }
        )

    @classmethod
    def _validate_durable_payload_inventory(cls, root: Path) -> None:
        expected = cls.durable_payload_filenames()
        observed = frozenset(path.name for path in root.iterdir())
        if observed != expected:
            raise MergeError(
                "T-Switch durable payload has unexpected entries: "
                f"expected={sorted(expected)} observed={sorted(observed)}"
            )
        invalid = sorted(
            path.name
            for path in root.iterdir()
            if not path.is_file() or path.is_symlink()
        )
        if invalid:
            raise MergeError(
                f"T-Switch durable payload entries must be regular files: {invalid}"
            )

    _DISCARD_RATIO_GROUPS = frozenset(
        {"attention", "mlp_or_expert", "norm", "other"}
    )

    def __init__(
        self,
        *,
        discard_ratio: float = 0.5,
        group_size: int = 4,
        discard_ratio_by_group: Mapping[str, float] | None = None,
    ) -> None:
        if not 0.0 <= discard_ratio < 1.0:
            raise MergeError("T-Switch discard_ratio must lie in [0, 1)")
        if group_size < 1:
            raise MergeError("T-Switch group_size must be positive")
        group_ratios = {
            str(key): float(value)
            for key, value in (discard_ratio_by_group or {}).items()
        }
        unknown_groups = sorted(set(group_ratios) - self._DISCARD_RATIO_GROUPS)
        if unknown_groups:
            raise MergeError(
                f"unknown T-Switch discard-ratio groups: {unknown_groups}"
            )
        invalid_groups = sorted(
            key for key, value in group_ratios.items() if not 0.0 <= value < 1.0
        )
        if invalid_groups:
            raise MergeError(
                "T-Switch group discard ratios must lie in [0, 1): "
                f"{invalid_groups}"
            )
        self.discard_ratio = float(discard_ratio)
        self.group_size = int(group_size)
        self.discard_ratio_by_group = group_ratios

    @staticmethod
    def parameter_group(key: str) -> str:
        """Map a parameter name to its transformer module group."""

        lower = key.lower()
        if any(
            token in lower
            for token in ("embed_tokens", "word_embeddings", "embedding")
        ):
            return "embedding"
        if "lm_head" in lower or "output_layer" in lower:
            return "lm_head"
        if "norm" in lower:
            return "norm"
        if any(
            token in lower
            for token in (
                "self_attn",
                "self_attention",
                ".attention.",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "query_key_value",
            )
        ):
            return "attention"
        if any(
            token in lower
            for token in (
                ".mlp.",
                ".experts.",
                "shared_expert",
                "gate_proj",
                "up_proj",
                "down_proj",
                ".router.",
            )
        ):
            return "mlp_or_expert"
        return "other"

    def discard_ratio_for(self, key: str) -> float:
        """Return a group-specific discard ratio or the default ratio."""

        return self.discard_ratio_by_group.get(
            self.parameter_group(key), self.discard_ratio
        )

    @staticmethod
    def _official_retained(delta: torch.Tensor, discard_ratio: float) -> torch.Tensor:
        """Select int(n * (1-alpha)) entries independently for each sign.

        Retain cutoff ties; a zero quota leaves that sign partition unchanged.
        """

        flat = delta.reshape(-1)
        retained = torch.zeros_like(flat)
        for positive in (True, False):
            selected = flat > 0 if positive else flat < 0
            values = flat[selected]
            if values.numel() == 0:
                continue
            magnitude = values if positive else values.abs()
            keep = int(values.numel() * (1.0 - discard_ratio))
            if keep > 0:
                # Find the magnitude cutoff with the (n-keep+1)-th order statistic.
                threshold_rank = magnitude.numel() - keep + 1
                threshold = torch.kthvalue(magnitude, threshold_rank).values
                keep_mask = selected & (
                    (flat >= threshold) if positive else (flat.abs() >= threshold)
                )
                if positive:
                    keep_mask &= flat > 0
                else:
                    keep_mask &= flat < 0
                retained[keep_mask] = flat[keep_mask]
            else:
                retained[selected] = flat[selected]
        return retained.reshape(delta.shape)

    def fit_and_serialize(
        self,
        base_source: TensorSource,
        endpoint_source: TensorSource,
        output_dir: str | Path,
        *,
        scale_granularity: str | Mapping[str, str] = "tensor",
        scale_method: str = "rms",
    ) -> TaskSwitchSerializedPayload:
        if base_source.keys != endpoint_source.keys:
            raise MergeError("T-Switch base and endpoint state dictionaries must align")
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        paths = {
            "group_mark": root / self._GROUP_MARK_FILE,
            "activation": root / self._ACTIVATION_FILE,
            "polarity": root / self._POLARITY_FILE,
            "scales": root / self._SCALES_FILE,
            "metadata": root / self._METADATA_FILE,
        }
        offsets = {"group_mark": 0, "activation": 0, "polarity": 0}
        scales: list[float] = []
        entries: list[dict[str, Any]] = []
        floating_parameter_count = 0
        active_parameter_count = 0
        task_vector_l2_squared = 0.0
        retained_task_vector_l2_squared = 0.0
        row_scaled_tensor_count = 0

        if scale_method not in {"rms", "ls"}:
            raise MergeError("T-Switch scale method must be rms or ls")

        if isinstance(scale_granularity, str):
            if scale_granularity not in {"tensor", "row", "row_for_rank_ge_2"}:
                raise MergeError(
                    "T-Switch scale granularity must be tensor, row, or "
                    "row_for_rank_ge_2"
                )

            def requested_granularity_for(_key: str) -> str:
                return scale_granularity

        else:
            granularities = {str(key): str(value) for key, value in scale_granularity.items()}
            invalid = sorted(
                key for key, value in granularities.items() if value not in {"tensor", "row"}
            )
            if invalid:
                raise MergeError(f"invalid T-Switch scale granularities: {invalid}")
            unknown = sorted(set(granularities) - set(base_source.keys))
            if unknown:
                raise MergeError(f"T-Switch scale map contains unknown tensors: {unknown}")

            def requested_granularity_for(key: str) -> str:
                return granularities.get(key, "tensor")

        def granularity_for(key: str, shape: tuple[int, ...]) -> str:
            requested = requested_granularity_for(key)
            if requested == "row_for_rank_ge_2":
                return "row" if len(shape) >= 2 else "tensor"
            return requested

        with (
            paths["group_mark"].open("wb") as mark_handle,
            paths["activation"].open("wb") as activation_handle,
            paths["polarity"].open("wb") as polarity_handle,
        ):
            for key in sorted(base_source.keys):
                base = base_source.tensor(key)
                endpoint = endpoint_source.tensor(key)
                if base.shape != endpoint.shape:
                    raise MergeError(f"T-Switch tensor shape mismatch for {key}")
                if not base.is_floating_point():
                    if not torch.equal(base, endpoint):
                        raise MergeError(f"non-floating T-Switch tensor differs: {key}")
                    entries.append(
                        {
                            "key": key,
                            "shape": list(base.shape),
                            "floating": False,
                            "source_dtype": str(base.dtype),
                        }
                    )
                    continue

                delta = endpoint.float() - base.float()
                retained = self._official_retained(delta, self.discard_ratio_for(key))
                flat_retained = retained.reshape(-1)
                active = flat_retained != 0
                positive = flat_retained > 0
                count = int(flat_retained.numel())
                groups = math.ceil(count / self.group_size)
                padded = groups * self.group_size

                active_np = np.zeros(padded, dtype=np.bool_)
                positive_np = np.zeros(padded, dtype=np.bool_)
                active_np[:count] = active.numpy()
                positive_np[:count] = positive.numpy()
                active_groups = active_np.reshape(groups, self.group_size)
                mark = active_groups.any(axis=1)
                activation_bits = active_groups[mark].reshape(-1)
                polarity_bits = positive_np[:count][active_np[:count]]

                mark_raw = np.packbits(mark, bitorder="little").tobytes()
                activation_raw = np.packbits(activation_bits, bitorder="little").tobytes()
                polarity_raw = np.packbits(polarity_bits, bitorder="little").tobytes()
                mark_handle.write(mark_raw)
                activation_handle.write(activation_raw)
                polarity_handle.write(polarity_raw)

                retained_norm = float(torch.linalg.vector_norm(flat_retained).item())
                active_count = int(active.sum().item())
                granularity = granularity_for(key, tuple(retained.shape))
                if granularity == "row":
                    if retained.ndim < 2:
                        raise MergeError(
                            f"row-scale T-Switch tensor must have rank at least two: {key}"
                        )
                    retained_rows = retained.reshape(retained.shape[0], -1)
                    active_rows = retained_rows != 0
                    row_counts = active_rows.sum(dim=1)
                    if scale_method == "rms":
                        numerator = retained_rows.square().sum(dim=1)
                        scale_tensor = torch.where(
                            row_counts > 0,
                            (numerator / row_counts.clamp_min(1).to(numerator.dtype)).sqrt(),
                            torch.zeros_like(numerator),
                        ).to(torch.float32)
                    else:
                        numerator = retained_rows.abs().sum(dim=1)
                        scale_tensor = torch.where(
                            row_counts > 0,
                            numerator / row_counts.clamp_min(1).to(numerator.dtype),
                            torch.zeros_like(numerator),
                        ).to(torch.float32)
                    scale_values = [float(value) for value in scale_tensor.tolist()]
                    row_scaled_tensor_count += 1
                else:
                    if scale_method == "rms":
                        scale = retained_norm / math.sqrt(active_count) if active_count else 0.0
                    else:
                        scale = (
                            float(flat_retained.abs().sum().item()) / active_count
                            if active_count
                            else 0.0
                        )
                    scale_values = [scale]
                scale_offset = len(scales)
                scales.extend(scale_values)
                entry = {
                    "key": key,
                    "shape": list(base.shape),
                    "floating": True,
                    "source_dtype": str(base.dtype),
                    "element_count": count,
                    "group_count": groups,
                    "active_group_count": int(mark.sum()),
                    "active_count": active_count,
                    "positive_active_count": int(positive.sum().item()),
                    "negative_active_count": int((flat_retained < 0).sum().item()),
                    "scale_index": scale_offset,
                    "scale": scale_values[0] if len(scale_values) == 1 else None,
                    "group_mark_offset": offsets["group_mark"],
                    "group_mark_bytes": len(mark_raw),
                    "activation_offset": offsets["activation"],
                    "activation_bytes": len(activation_raw),
                    "activation_bit_count": int(activation_bits.size),
                    "polarity_offset": offsets["polarity"],
                    "polarity_bytes": len(polarity_raw),
                    "polarity_bit_count": active_count,
                }
                entries.append(entry)
                offsets["group_mark"] += len(mark_raw)
                offsets["activation"] += len(activation_raw)
                offsets["polarity"] += len(polarity_raw)
                floating_parameter_count += count
                active_parameter_count += active_count
                task_vector_l2_squared += float(torch.sum(delta.double().square()).item())
                retained_task_vector_l2_squared += retained_norm * retained_norm

        paths["scales"].write_bytes(struct.pack(f"<{len(scales)}f", *scales) if scales else b"")
        schema_version = 2 if row_scaled_tensor_count else 1
        if schema_version == 2:
            scale_offset = 0
            for entry in entries:
                if not bool(entry.get("floating", False)):
                    continue
                granularity = granularity_for(
                    str(entry["key"]), tuple(int(value) for value in entry["shape"])
                )
                scale_count = int(entry["shape"][0]) if granularity == "row" else 1
                entry["scale"] = None if granularity == "row" else entry["scale"]
                entry["scale_index"] = scale_offset
                entry["scale_offset_f32"] = scale_offset
                entry["scale_count"] = scale_count
                entry["scale_granularity"] = granularity
                scale_offset += scale_count
        metadata: dict[str, Any] = {
            "schema_version": schema_version,
            "method": (
                "T-Switch-row-scale-extension"
                if schema_version == 2
                else "T-Switch"
            ),
            "paper": "https://arxiv.org/abs/2412.00054",
            "official_repository": "https://github.com/lfy-123/Binary-Task-Switch",
            "official_repository_revision": "68bfda8f67ad4cc7293bd6564e8064fd5c536551",
            "discard_ratio": self.discard_ratio,
            "discard_ratio_by_group": dict(self.discard_ratio_by_group),
            "group_size": self.group_size,
            "scale_method": scale_method,
            "scope": "all_aligned_floating_tensors_including_lm_head",
            "generative_lm_extension": (
                "the reference RoBERTa implementation excludes its task-specific classifier; "
                "this homologous generative-LM control has no separately stored classifier and "
                "therefore switches every aligned floating tensor including lm_head.weight"
            ),
            "official_tensor_semantics": {
                "discard": (
                    "per named tensor; positive and negative entries ranked separately; "
                    "retain int(sign_count*(1-alpha)) plus threshold ties"
                ),
                "polarity": "sign(retained_task_vector)",
                "scale": (
                    "sqrt(mean(square(retained_task_vector_on_active_entries)))"
                    if scale_method == "rms"
                    else "mean(abs(retained_task_vector_on_active_entries))"
                ),
                "reconstruction": "base + scale_t * activation_t * polarity_t",
            },
            "serialization": (
                "grouped equivalent of the official group-size bitarray: one group mark, "
                "activation bits for marked groups, polarity bits for active entries"
            ),
            "bitorder": "little",
            "output_dtype": "bfloat16",
            "floating_parameter_count": floating_parameter_count,
            "active_parameter_count": active_parameter_count,
            "active_density": (
                active_parameter_count / floating_parameter_count
                if floating_parameter_count
                else 0.0
            ),
            "task_vector_l2": math.sqrt(task_vector_l2_squared),
            "retained_task_vector_l2": math.sqrt(retained_task_vector_l2_squared),
            "tensor_scale_count": (
                len(scales)
                if schema_version == 1
                else sum(bool(entry.get("floating", False)) for entry in entries)
            ),
            "entries": entries,
        }
        if schema_version == 2:
            metadata["scale_value_count"] = len(scales)
            metadata["row_scaled_tensor_count"] = row_scaled_tensor_count
            metadata["scale_extension"] = {
                "name": "output-row scale extension",
                "official_t_switch": False,
                "scale_method": scale_method,
                "row_semantics": "one fp32 scale per output row / first dimension",
                "reconstruction": "base + broadcast(scale) * activation * polarity",
            }
        paths["metadata"].write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._validate_durable_payload_inventory(root)
        file_bytes = {name: path.stat().st_size for name, path in paths.items()}
        file_sha256 = {name: _sha256(path) for name, path in paths.items()}
        return TaskSwitchSerializedPayload(root, metadata, file_bytes, file_sha256)

    def rescale_serialized(
        self,
        base_source: TensorSource,
        endpoint_source: TensorSource,
        reference_payload: TaskSwitchSerializedPayload,
        output_dir: str | Path,
        *,
        scale_method: str = "ls",
        scale_granularity: str = "row_for_rank_ge_2",
        row_weight_by_key: Mapping[str, torch.Tensor] | None = None,
        scale_provenance: Mapping[str, Any] | None = None,
        frozen_file_strategy: str = "copy",
    ) -> TaskSwitchSerializedPayload:
        """Recompute tensor or row scales using the saved support and sign bits."""

        if scale_method not in {"ls", "activation_ls"} or scale_granularity != "row_for_rank_ge_2":
            raise MergeError(
                "frozen-bit rescale supports LS-row or activation-weighted LS-row"
            )
        if frozen_file_strategy not in {"copy", "hardlink"}:
            raise MergeError("frozen-file strategy must be copy or hardlink")
        row_weights = {
            str(key): value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            for key, value in (row_weight_by_key or {}).items()
        }
        if scale_method == "activation_ls" and not row_weights:
            raise MergeError("activation-weighted LS-row requires frozen input moments")
        if scale_method == "ls" and row_weights:
            raise MergeError("ordinary LS-row cannot consume activation moments")
        if base_source.keys != endpoint_source.keys:
            raise MergeError("T-Switch base and endpoint state dictionaries must align")
        if float(reference_payload.metadata.get("discard_ratio", -1.0)) != self.discard_ratio:
            raise MergeError("reference T-Switch discard ratio changed")
        entries = json.loads(json.dumps(reference_payload.metadata.get("entries", [])))
        entry_by_key = {str(entry["key"]): entry for entry in entries}
        if frozenset(entry_by_key) != base_source.keys:
            raise MergeError("reference T-Switch tensor inventory changed")

        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        for filename in (
            self._GROUP_MARK_FILE,
            self._ACTIVATION_FILE,
            self._POLARITY_FILE,
        ):
            source = reference_payload.root / filename
            destination = root / filename
            if frozen_file_strategy == "hardlink":
                os.link(source, destination)
            else:
                shutil.copyfile(source, destination)

        scales: list[float] = []
        row_scaled_tensor_count = 0
        for key in sorted(base_source.keys):
            entry = entry_by_key[key]
            base = base_source.tensor(key)
            endpoint = endpoint_source.tensor(key)
            if list(base.shape) != list(entry["shape"]) or base.shape != endpoint.shape:
                raise MergeError(f"frozen-bit T-Switch tensor shape changed: {key}")
            if not bool(entry.get("floating", False)):
                if not torch.equal(base, endpoint):
                    raise MergeError(f"non-floating T-Switch tensor differs: {key}")
                continue

            bits = self.decode_tensor_bits(key=key, payload=reference_payload)
            count = int(entry["element_count"])
            shifts = torch.arange(8, dtype=torch.uint8)
            active = (
                ((bits.activation_bitpack[:, None] >> shifts[None, :]) & 1)
                .reshape(-1)[:count]
                .bool()
            )
            positive = (
                ((bits.polarity_bitpack[:, None] >> shifts[None, :]) & 1)
                .reshape(-1)[:count]
                .bool()
            )
            signed_code = torch.where(positive, 1.0, -1.0) * active.float()
            delta = (endpoint.float() - base.float()).reshape(-1)
            if torch.any((delta * signed_code)[active] < 0):
                raise MergeError(f"reference T-Switch polarity changed: {key}")

            if delta.ndim != 1:
                raise AssertionError("flattened T-Switch delta must be one-dimensional")
            if base.ndim >= 2:
                delta_rows = delta.reshape(base.shape[0], -1)
                code_rows = signed_code.reshape(base.shape[0], -1)
                active_rows = active.reshape(base.shape[0], -1)
                weight = row_weights.get(key)
                if weight is not None:
                    if weight.numel() != delta_rows.shape[1] or torch.any(weight < 0):
                        raise MergeError(
                            f"activation moment geometry changed for {key}"
                        )
                    if not torch.isfinite(weight).all():
                        raise MergeError(f"activation moments are non-finite for {key}")
                    weight = weight.to(dtype=delta_rows.dtype).reshape(1, -1)
                    counts = (active_rows.to(delta_rows.dtype) * weight).sum(dim=1)
                    numerator = (delta_rows * code_rows * weight).sum(dim=1)
                else:
                    counts = active_rows.sum(dim=1).to(delta_rows.dtype)
                    numerator = (delta_rows * code_rows).sum(dim=1)
                scale_tensor = torch.where(
                    counts > 0,
                    numerator / counts.clamp_min(torch.finfo(numerator.dtype).tiny),
                    torch.zeros_like(numerator),
                ).to(torch.float32)
                scale_values = [float(value) for value in scale_tensor.tolist()]
                granularity = "row"
                row_scaled_tensor_count += 1
            else:
                active_count = int(active.sum().item())
                scale_values = [
                    float((delta * signed_code).sum().item()) / active_count
                    if active_count
                    else 0.0
                ]
                granularity = "tensor"

            offset = len(scales)
            scales.extend(scale_values)
            entry["scale"] = scale_values[0] if granularity == "tensor" else None
            entry["scale_index"] = offset
            entry["scale_offset_f32"] = offset
            entry["scale_count"] = len(scale_values)
            entry["scale_granularity"] = granularity

        (root / self._SCALES_FILE).write_bytes(
            struct.pack(f"<{len(scales)}f", *scales) if scales else b""
        )
        metadata = dict(reference_payload.metadata)
        metadata.update(
            {
                "schema_version": 2,
                "method": "T-Switch-row-scale-extension",
                "scale_method": scale_method,
                "entries": entries,
                "tensor_scale_count": sum(
                    bool(entry.get("floating", False)) for entry in entries
                ),
                "scale_value_count": len(scales),
                "row_scaled_tensor_count": row_scaled_tensor_count,
                "scale_extension": {
                    "name": "output-row scale extension",
                    "official_t_switch": False,
                    "scale_method": scale_method,
                    "row_semantics": "one fp32 scale per output row / first dimension",
                    "reconstruction": "base + broadcast(scale) * activation * polarity",
                    "frozen_bit_sha256": {
                        filename: reference_payload.file_sha256[name]
                        for name, filename in (
                            ("group_mark", self._GROUP_MARK_FILE),
                            ("activation", self._ACTIVATION_FILE),
                            ("polarity", self._POLARITY_FILE),
                        )
                    },
                    "activation_weighted_keys": sorted(row_weights),
                    "scale_provenance": dict(scale_provenance or {}),
                    "frozen_file_strategy": frozen_file_strategy,
                },
            }
        )
        semantics = dict(metadata.get("official_tensor_semantics", {}))
        semantics["scale"] = (
            "diagonal_activation_weighted_least_squares_fixed_signed_code"
            if scale_method == "activation_ls"
            else "least_squares_fixed_signed_code"
        )
        metadata["official_tensor_semantics"] = semantics
        (root / self._METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._validate_durable_payload_inventory(root)
        paths = {
            "group_mark": root / self._GROUP_MARK_FILE,
            "activation": root / self._ACTIVATION_FILE,
            "polarity": root / self._POLARITY_FILE,
            "scales": root / self._SCALES_FILE,
            "metadata": root / self._METADATA_FILE,
        }
        file_bytes = {name: path.stat().st_size for name, path in paths.items()}
        file_sha256 = {name: _sha256(path) for name, path in paths.items()}
        return TaskSwitchSerializedPayload(root, metadata, file_bytes, file_sha256)

    @classmethod
    def load_serialized(
        cls,
        root: str | Path,
        *,
        expected_file_sha256: Mapping[str, str],
        expected_file_bytes: Mapping[str, int],
        allow_schema_v2: bool = False,
    ) -> TaskSwitchSerializedPayload:
        payload_root = Path(root).expanduser().resolve()
        cls._validate_durable_payload_inventory(payload_root)
        paths = {
            "group_mark": payload_root / cls._GROUP_MARK_FILE,
            "activation": payload_root / cls._ACTIVATION_FILE,
            "polarity": payload_root / cls._POLARITY_FILE,
            "scales": payload_root / cls._SCALES_FILE,
            "metadata": payload_root / cls._METADATA_FILE,
        }
        if set(expected_file_sha256) != set(paths) or set(expected_file_bytes) != set(paths):
            raise MergeError("T-Switch payload manifest has an incomplete inventory")
        observed_bytes = {name: path.stat().st_size for name, path in paths.items()}
        observed_hashes = {name: _sha256(path) for name, path in paths.items()}
        if observed_bytes != {name: int(value) for name, value in expected_file_bytes.items()}:
            raise MergeError("T-Switch payload byte inventory changed")
        if observed_hashes != dict(expected_file_sha256):
            raise MergeError("T-Switch payload hash inventory changed")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        schema_version = int(metadata.get("schema_version", 0))
        valid_v1 = schema_version == 1 and metadata.get("method") == "T-Switch"
        valid_v2 = (
            allow_schema_v2
            and schema_version == 2
            and metadata.get("method") == "T-Switch-row-scale-extension"
        )
        if not (valid_v1 or valid_v2):
            raise MergeError("invalid T-Switch payload metadata")
        if valid_v2:
            scale_values = int(metadata.get("scale_value_count", -1))
            if scale_values < 0 or observed_bytes["scales"] != 4 * scale_values:
                raise MergeError("T-Switch v2 scale inventory changed")
            next_scale_offset = 0
            for entry in metadata.get("entries", []):
                if not bool(entry.get("floating", False)):
                    continue
                granularity = str(entry.get("scale_granularity", ""))
                count = int(entry.get("scale_count", -1))
                shape = tuple(int(value) for value in entry.get("shape", []))
                expected = (
                    1
                    if granularity == "tensor"
                    else (shape[0] if len(shape) >= 2 else -1)
                )
                offset = int(entry.get("scale_offset_f32", -1))
                if granularity not in {"tensor", "row"} or count != expected:
                    raise MergeError("T-Switch v2 scale geometry changed")
                if offset != next_scale_offset or offset + count > scale_values:
                    raise MergeError("T-Switch v2 scale offset changed")
                next_scale_offset += count
            if next_scale_offset != scale_values:
                raise MergeError("T-Switch v2 scale coverage changed")
        return TaskSwitchSerializedPayload(payload_root, metadata, observed_bytes, observed_hashes)

    @staticmethod
    def _entry_map(payload: TaskSwitchSerializedPayload) -> dict[str, Mapping[str, Any]]:
        entries = payload.metadata.get("entries")
        if not isinstance(entries, list):
            raise MergeError("T-Switch payload entries are missing")
        result = {str(entry["key"]): entry for entry in entries}
        if len(result) != len(entries):
            raise MergeError("T-Switch payload contains duplicate tensor keys")
        return result

    @classmethod
    def decode_tensor_bits(
        cls,
        *,
        key: str,
        payload: TaskSwitchSerializedPayload,
    ) -> TaskSwitchTensorBits:
        """Decode one compact tensor entry into random-access support and sign bits."""

        entry = cls._entry_map(payload).get(key)
        if entry is None:
            raise MergeError(f"T-Switch payload does not contain tensor: {key}")
        shape = tuple(int(value) for value in entry["shape"])
        if not bool(entry["floating"]):
            raise MergeError(f"T-Switch tensor is not floating: {key}")
        count = int(entry["element_count"])
        group_size = int(payload.metadata["group_size"])
        groups = int(entry["group_count"])
        with (
            (payload.root / cls._GROUP_MARK_FILE).open("rb") as mark_handle,
            (payload.root / cls._ACTIVATION_FILE).open("rb") as activation_handle,
            (payload.root / cls._POLARITY_FILE).open("rb") as polarity_handle,
        ):
            mark_handle.seek(int(entry["group_mark_offset"]))
            mark = np.unpackbits(
                np.frombuffer(
                    mark_handle.read(int(entry["group_mark_bytes"])), dtype=np.uint8
                ),
                bitorder="little",
                count=groups,
            ).astype(np.bool_)
            activation_handle.seek(int(entry["activation_offset"]))
            activation_bits = np.unpackbits(
                np.frombuffer(
                    activation_handle.read(int(entry["activation_bytes"])),
                    dtype=np.uint8,
                ),
                bitorder="little",
                count=int(entry["activation_bit_count"]),
            ).astype(np.bool_)
            active = np.zeros((groups, group_size), dtype=np.bool_)
            active[mark] = activation_bits.reshape(int(mark.sum()), group_size)
            active = active.reshape(-1)[:count]
            polarity_handle.seek(int(entry["polarity_offset"]))
            active_polarity = np.unpackbits(
                np.frombuffer(
                    polarity_handle.read(int(entry["polarity_bytes"])), dtype=np.uint8
                ),
                bitorder="little",
                count=int(entry["polarity_bit_count"]),
            ).astype(np.bool_)
        polarity = np.zeros(count, dtype=np.bool_)
        polarity[active] = active_polarity
        # Copy little-endian unpacked bits into writable tensor storage.
        activation_packed = torch.from_numpy(
            np.packbits(active, bitorder="little").copy()
        ).to(torch.uint8)
        polarity_packed = torch.from_numpy(
            np.packbits(polarity, bitorder="little").copy()
        ).to(torch.uint8)
        schema_version = int(payload.metadata.get("schema_version", 1))
        if schema_version == 1:
            scale_values = torch.tensor([float(entry["scale"])], dtype=torch.float32)
            scale_granularity = "tensor"
        elif schema_version == 2:
            scale_granularity = str(entry["scale_granularity"])
            scale_count = int(entry["scale_count"])
            with (payload.root / cls._SCALES_FILE).open("rb") as scale_handle:
                scale_handle.seek(4 * int(entry["scale_offset_f32"]))
                raw = scale_handle.read(4 * scale_count)
            if len(raw) != 4 * scale_count:
                raise MergeError(f"T-Switch row scales are truncated for {key}")
            scale_values = torch.from_numpy(
                np.frombuffer(raw, dtype="<f4", count=scale_count).copy()
            ).to(torch.float32)
        else:
            raise MergeError("unsupported T-Switch payload schema")
        return TaskSwitchTensorBits(
            key=key,
            shape=shape,
            activation_bitpack=activation_packed,
            polarity_bitpack=polarity_packed,
            scale=float(scale_values[0]),
            scale_values=scale_values,
            scale_granularity=scale_granularity,
        )

    @classmethod
    def reconstruct_tensor(
        cls,
        *,
        key: str,
        base: torch.Tensor,
        payload: TaskSwitchSerializedPayload,
    ) -> torch.Tensor:
        entry = cls._entry_map(payload).get(key)
        if entry is None or list(base.shape) != list(entry["shape"]):
            raise MergeError(f"T-Switch payload/base mismatch for {key}")
        if not bool(entry["floating"]):
            return base.clone()
        count = int(entry["element_count"])
        switch = cls.decode_tensor_bits(key=key, payload=payload)
        shifts = torch.arange(8, dtype=torch.uint8)
        active_t = (
            (switch.activation_bitpack[:, None] >> shifts[None, :]) & 1
        ).reshape(-1)[:count].bool()
        positive_t = (
            (switch.polarity_bitpack[:, None] >> shifts[None, :]) & 1
        ).reshape(-1)[:count].bool()
        sign = torch.where(positive_t, 1.0, -1.0) * active_t.float()
        signed = sign.reshape(base.shape)
        reconstructed = base.float() + switch.broadcast_scale(device=base.device) * signed
        return reconstructed.to(torch.bfloat16)

    def apply_serialized(
        self,
        *,
        destination: Mapping[str, torch.Tensor],
        base_source: TensorSource,
        payload: TaskSwitchSerializedPayload,
        expected_source: TensorSource | None = None,
    ) -> TaskSwitchApplicationReceipt:
        if frozenset(destination) != base_source.keys:
            raise MergeError("T-Switch destination/base key mismatch")
        if expected_source is not None and expected_source.keys != base_source.keys:
            raise MergeError("T-Switch expected endpoint/base key mismatch")
        digest = hashlib.sha256()
        count = 0
        exact = True if expected_source is not None else None
        max_error = 0.0 if expected_source is not None else None
        squared_error = 0.0
        error_count = 0
        with torch.no_grad():
            for key in sorted(base_source.keys):
                tensor = self.reconstruct_tensor(
                    key=key, base=base_source.tensor(key), payload=payload
                )
                destination[key].copy_(
                    tensor.to(device=destination[key].device, dtype=destination[key].dtype)
                )
                digest.update(
                    json.dumps(
                        [key, str(tensor.dtype), tensor.numel()], separators=(",", ":")
                    ).encode()
                    + b"\0"
                )
                digest.update(tensor.contiguous().view(torch.uint8).numpy())
                count += tensor.numel()
                if expected_source is not None:
                    expected = expected_source.tensor(key).to(torch.bfloat16)
                    difference = tensor.float() - expected.float()
                    if not torch.equal(tensor, expected):
                        exact = False
                    max_error = max(max_error or 0.0, float(difference.abs().max().item()))
                    squared_error += float(torch.sum(difference.double().square()).item())
                    error_count += difference.numel()
        return TaskSwitchApplicationReceipt(
            tensor_sha256=digest.hexdigest(),
            materialized_parameter_count=count,
            endpoint_exact=exact,
            max_abs_error=max_error,
            mean_squared_error=(squared_error / error_count if error_count else None),
        )


class TaskSwitchDenseCheckpointPreparer:
    """Materialize a dense checkpoint from a compact task switch.

    The no_lm_head policy copies the endpoint output head; no_embed_lm_head
    copies both endpoint embedding and head. Other tensors use the switch payload.
    """

    HEAD_KEY = "lm_head.weight"
    EMBEDDING_KEY = "model.embed_tokens.weight"
    HEAD_POLICIES = frozenset({"full", "no_lm_head", "no_embed_lm_head"})
    ROUTED_KEYS = {
        "full": frozenset(),
        "no_lm_head": frozenset({HEAD_KEY}),
        "no_embed_lm_head": frozenset({EMBEDDING_KEY, HEAD_KEY}),
    }

    def __init__(
        self,
        *,
        base_source: TensorSource,
        endpoint_source: TensorSource,
        endpoint_snapshot: str | Path,
        payload: TaskSwitchSerializedPayload,
        head_policy: str,
        head_key: str = HEAD_KEY,
        embedding_key: str = EMBEDDING_KEY,
    ) -> None:
        if head_policy not in self.HEAD_POLICIES:
            raise MergeError(f"unsupported T-Switch head policy: {head_policy}")
        if base_source.keys != endpoint_source.keys:
            raise MergeError("T-Switch dense endpoint/base keys do not align")
        if base_source.keys != frozenset(
            str(entry["key"]) for entry in payload.metadata["entries"]
        ):
            raise MergeError("T-Switch payload/base keys do not align")
        if not head_key or not embedding_key or head_key == embedding_key:
            raise MergeError("T-Switch routed tensor keys must be distinct and non-empty")
        routed_keys = {
            "full": frozenset(),
            "no_lm_head": frozenset({head_key}),
            "no_embed_lm_head": frozenset({embedding_key, head_key}),
        }
        missing_routed = routed_keys[head_policy] - base_source.keys
        if missing_routed:
            raise MergeError(
                f"{head_policy} requires routed tensors: {sorted(missing_routed)}"
            )
        self.base_source = base_source
        self.endpoint_source = endpoint_source
        self.endpoint_snapshot = Path(endpoint_snapshot).expanduser().resolve()
        self.payload = payload
        self.head_policy = head_policy
        self.head_key = head_key
        self.embedding_key = embedding_key
        self.routed_keys = routed_keys

    def prepare(self, destination: str | Path, *, identity: Mapping[str, Any],
                consumer_tasks: tuple[str, ...] | None = None) -> dict[str, Any]:
        if consumer_tasks is not None and (not consumer_tasks or len(set(consumer_tasks)) != len(consumer_tasks)):
            raise ValueError("consumer tasks must be nonempty and unique")
        output = Path(destination).expanduser().resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        )
        try:
            index_path = self.endpoint_snapshot / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = {str(key): str(shard) for key, shard in dict(index["weight_map"]).items()}
            if frozenset(weight_map) != self.base_source.keys:
                raise MergeError("endpoint index changed after T-Switch key validation")
            keys_by_shard: dict[str, list[str]] = defaultdict(list)
            for key, shard in weight_map.items():
                keys_by_shard[shard].append(key)
            self._copy_non_weight_files(temporary)

            shards: dict[str, dict[str, Any]] = {}
            tensor_count = 0
            parameter_count = 0
            squared_error = 0.0
            error_count = 0
            max_abs_error = 0.0
            for shard in sorted(keys_by_shard):
                tensors: dict[str, torch.Tensor] = {}
                for key in sorted(keys_by_shard[shard]):
                    endpoint = self.endpoint_source.tensor(key)
                    if key in self.routed_keys[self.head_policy]:
                        tensor = endpoint.to(torch.bfloat16).contiguous()
                    else:
                        tensor = TaskSwitchBuilder.reconstruct_tensor(
                            key=key,
                            base=self.base_source.tensor(key),
                            payload=self.payload,
                        ).contiguous()
                    tensors[key] = tensor
                    tensor_count += 1
                    parameter_count += tensor.numel()
                    if tensor.is_floating_point():
                        difference = tensor.float() - endpoint.to(torch.bfloat16).float()
                        squared_error += float(torch.sum(difference.double().square()).item())
                        error_count += difference.numel()
                        max_abs_error = max(max_abs_error, float(difference.abs().max().item()))
                shard_path = temporary / shard
                save_file(tensors, str(shard_path), metadata={"format": "pt"})
                shards[shard] = {
                    "sha256": _sha256(shard_path),
                    "size_bytes": shard_path.stat().st_size,
                    "tensor_count": len(tensors),
                }
                del tensors
                _atomic_json(
                    temporary / "PREPARATION_PROGRESS.json",
                    {
                        "status": "RUNNING",
                        "completed_shards": sorted(shards),
                        "tensor_count": tensor_count,
                        "parameter_count": parameter_count,
                    },
                )

            output_index = {
                "metadata": {"total_size": sum(row["size_bytes"] for row in shards.values())},
                "weight_map": weight_map,
            }
            _atomic_json(temporary / "model.safetensors.index.json", output_index)
            receipt: dict[str, Any] = {
                "schema_version": 1,
                "status": "PREPARED",
                "prepared_at": datetime.now(UTC).isoformat(),
                **dict(identity),
                "method": "T-Switch",
                "discard_ratio": float(self.payload.metadata["discard_ratio"]),
                "discard_ratio_by_group": dict(
                    self.payload.metadata.get("discard_ratio_by_group", {})
                ),
                "group_size": int(self.payload.metadata["group_size"]),
                "head_policy": self.head_policy,
                "head_semantics": (
                    "switched_with_body"
                    if self.head_policy == "full"
                    else (
                        "exact_mode_native_endpoint_head"
                        if self.head_policy == "no_lm_head"
                        else "exact_mode_native_endpoint_embedding_and_head"
                    )
                ),
                "mode_native_routed_keys": sorted(self.routed_keys[self.head_policy]),
                "head_key": self.head_key,
                "embedding_key": self.embedding_key,
                "payload_root": str(self.payload.root),
                "payload_file_sha256": dict(self.payload.file_sha256),
                "payload_file_bytes": dict(self.payload.file_bytes),
                "endpoint_snapshot": str(self.endpoint_snapshot),
                "endpoint_index_sha256": _sha256(index_path),
                "index_sha256": _sha256(temporary / "model.safetensors.index.json"),
                "tensor_count": tensor_count,
                "parameter_count": parameter_count,
                "max_abs_error_against_endpoint_bf16": max_abs_error,
                "mean_squared_error_against_endpoint_bf16": (
                    squared_error / error_count if error_count else None
                ),
                "shards": shards,
                "durable_storage_policy": "compact_switch_payload_only",
                "dense_checkpoint_lifecycle": "queue_scoped_until_last_gpu_consumer_terminal",
                "consumer_contract": ("shared_read_only_candidate_mode_three_tasks" if consumer_tasks is None
                                      else "shared_read_only_candidate_mode_atomic_tasks"),
                "expected_full_consumer_count": len(consumer_tasks) if consumer_tasks is not None else 3,
                "expected_canary_consumer_count": 1,
                "expected_total_consumer_count": 1 + len(consumer_tasks) if consumer_tasks is not None else 4,
            }
            if consumer_tasks is not None:
                receipt["consumer_tasks"] = list(consumer_tasks)
            _atomic_json(temporary / "PREPARED.json", receipt)
            (temporary / "PREPARATION_PROGRESS.json").unlink(missing_ok=True)
            temporary.replace(output)
            return receipt
        except BaseException as exc:
            _atomic_json(
                temporary / "FAILED.json",
                {
                    "status": "FAILED",
                    "failed_at": datetime.now(UTC).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def _copy_non_weight_files(self, destination: Path) -> None:
        for source in self.endpoint_snapshot.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(self.endpoint_snapshot)
            if relative.name == "model.safetensors.index.json" or relative.name.endswith(
                ".safetensors"
            ):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
