"""Training-free Model Soup / Task Arithmetic tensor interpolation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


class MergeError(ValueError):
    """Exception for incompatible endpoint tensors or merge configurations."""


class TensorSource(Protocol):
    """Interface for random-access checkpoint tensors."""

    @property
    def keys(self) -> frozenset[str]:
        """Return the complete state-dict key set."""

    def tensor(self, key: str) -> torch.Tensor:
        """Load one CPU tensor by canonical state-dict key."""


class MappingTensorSource:
    """Tensor source backed by an in-memory mapping."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self._tensors = tensors

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self._tensors)

    def tensor(self, key: str) -> torch.Tensor:
        return self._tensors[key]


class RestrictedTensorSource:
    """Expose a validated subset of keys from another tensor source."""

    def __init__(self, source: TensorSource, keys: frozenset[str]) -> None:
        missing = keys - source.keys
        if missing:
            raise MergeError(f"restricted tensor source is missing keys: {sorted(missing)[:8]}")
        self._source = source
        self._keys = frozenset(keys)

    @property
    def keys(self) -> frozenset[str]:
        return self._keys

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self._keys:
            raise KeyError(key)
        return self._source.tensor(key)


class SafetensorCheckpointSource:
    """Memory-map safetensor shards and retrieve tensors by state-dict key."""

    def __init__(self, snapshot: str | Path) -> None:
        self.snapshot = Path(snapshot).expanduser().resolve()
        index_path = self.snapshot / "model.safetensors.index.json"
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            raw_weight_map = loaded["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
            raise MergeError(f"invalid safetensor index: {index_path}") from exc
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise MergeError(f"empty safetensor weight map: {index_path}")
        self._weight_map = {str(key): str(value) for key, value in raw_weight_map.items()}
        self._stack: ExitStack | None = None
        self._handles: dict[str, Any] = {}

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self._weight_map)

    def __enter__(self) -> SafetensorCheckpointSource:
        from safetensors import safe_open

        if self._stack is not None:
            raise MergeError("safetensor source is already open")
        self._stack = ExitStack()
        for shard_name in sorted(set(self._weight_map.values())):
            shard_path = self.snapshot / shard_name
            if not shard_path.is_file():
                self._stack.close()
                self._stack = None
                raise MergeError(f"missing safetensor shard: {shard_path}")
            self._handles[shard_name] = self._stack.enter_context(
                safe_open(shard_path, framework="pt", device="cpu")
            )
        return self

    def __exit__(self, *_: Any) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles.clear()

    def tensor(self, key: str) -> torch.Tensor:
        if self._stack is None:
            raise MergeError("safetensor source must be opened as a context manager")
        shard_name = self._weight_map[key]
        return self._handles[shard_name].get_tensor(key)

    def shape(self, key: str) -> tuple[int, ...]:
        """Read one tensor shape from safetensors metadata without materializing it."""

        if self._stack is None:
            raise MergeError("safetensor source must be opened as a context manager")
        shard_name = self._weight_map[key]
        return tuple(int(value) for value in self._handles[shard_name].get_slice(key).get_shape())


@dataclass(frozen=True, slots=True)
class LinearMergeTensorConfig:
    """Interpolation coefficient and output dtype for a linear merge."""

    lambda_value: float
    accumulation_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_value <= 1.0:
            raise MergeError("lambda_value must lie in [0, 1]")
        if self.accumulation_dtype is not torch.float32:
            raise MergeError("the pilot requires float32 accumulation")


class LinearMergeStrategy:
    """Interpolate aligned tensors in FP32 and cast to the configured output dtype."""

    def __init__(self, config: LinearMergeTensorConfig) -> None:
        self.config = config

    def merge_tensor(self, ar_tensor: torch.Tensor, diffusion_tensor: torch.Tensor) -> torch.Tensor:
        """Merge two aligned tensors and cast the FP32 result to the output dtype."""

        if ar_tensor.shape != diffusion_tensor.shape:
            raise MergeError(
                f"tensor shape mismatch: AR {tuple(ar_tensor.shape)} vs "
                f"diffusion {tuple(diffusion_tensor.shape)}"
            )
        if ar_tensor.is_floating_point() != diffusion_tensor.is_floating_point():
            raise MergeError("floating/non-floating tensor kind mismatch")

        if not ar_tensor.is_floating_point():
            if not torch.equal(ar_tensor, diffusion_tensor):
                raise MergeError("non-floating endpoint tensors must agree exactly")
            return ar_tensor.clone()

        # Return source tensors directly at interpolation coefficients zero and one.
        if self.config.lambda_value == 0.0:
            return ar_tensor.to(dtype=self.config.output_dtype).clone()
        if self.config.lambda_value == 1.0:
            return diffusion_tensor.to(dtype=self.config.output_dtype).clone()

        ar_acc = ar_tensor.to(dtype=self.config.accumulation_dtype)
        diffusion_acc = diffusion_tensor.to(dtype=self.config.accumulation_dtype)
        merged = ar_acc + self.config.lambda_value * (diffusion_acc - ar_acc)
        return merged.to(dtype=self.config.output_dtype)

    def iter_merged_tensors(
        self,
        ar_tensors: Mapping[str, torch.Tensor],
        diffusion_tensors: Mapping[str, torch.Tensor],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield merged tensors one at a time."""

        ar_keys = set(ar_tensors)
        diffusion_keys = set(diffusion_tensors)
        if ar_keys != diffusion_keys:
            missing_from_diffusion = sorted(ar_keys - diffusion_keys)
            missing_from_ar = sorted(diffusion_keys - ar_keys)
            raise MergeError(
                "state-dict keys differ; "
                f"missing_from_diffusion={missing_from_diffusion[:8]}, "
                f"missing_from_ar={missing_from_ar[:8]}"
            )

        for key in sorted(ar_keys):
            yield key, self.merge_tensor(ar_tensors[key], diffusion_tensors[key])


@dataclass(frozen=True, slots=True)
class MaterializationReceipt:
    """Merged tensor hashes, dtypes, and parameter counts."""

    tensor_sha256: str
    tensor_count: int
    parameter_count: int
    endpoint_exact: bool | None


class TransientLinearModelMaterializer:
    """Load merged tensors into an existing model one tensor at a time."""

    def __init__(self, strategy: LinearMergeStrategy) -> None:
        self._strategy = strategy

    def materialize(
        self,
        destination: Mapping[str, torch.Tensor],
        ar_source: TensorSource,
        diffusion_source: TensorSource,
    ) -> MaterializationReceipt:
        destination_keys = frozenset(destination)
        if destination_keys != ar_source.keys or destination_keys != diffusion_source.keys:
            raise MergeError(
                "destination and endpoint state-dict keys must agree exactly; "
                f"destination={len(destination_keys)}, ar={len(ar_source.keys)}, "
                f"diffusion={len(diffusion_source.keys)}"
            )

        digest = hashlib.sha256()
        parameter_count = 0
        endpoint_exact: bool | None = (
            True if self._strategy.config.lambda_value in {0.0, 1.0} else None
        )
        with torch.no_grad():
            for key in sorted(destination_keys):
                ar_tensor = ar_source.tensor(key)
                diffusion_tensor = diffusion_source.tensor(key)
                merged = self._strategy.merge_tensor(ar_tensor, diffusion_tensor).contiguous()
                target = destination[key]
                if target.shape != merged.shape:
                    raise MergeError(
                        f"destination shape mismatch for {key}: "
                        f"{tuple(target.shape)} vs {tuple(merged.shape)}"
                    )
                if target.dtype != merged.dtype:
                    raise MergeError(
                        f"destination dtype mismatch for {key}: {target.dtype} vs {merged.dtype}"
                    )

                header = json.dumps(
                    [key, str(merged.dtype), list(merged.shape)],
                    separators=(",", ":"),
                ).encode("utf-8")
                digest.update(header + b"\0")
                digest.update(merged.view(torch.uint8).numpy())
                target.copy_(merged, non_blocking=False)
                parameter_count += merged.numel()

                if endpoint_exact:
                    expected = (
                        ar_tensor if self._strategy.config.lambda_value == 0.0 else diffusion_tensor
                    )
                    endpoint_exact = torch.equal(
                        merged, expected.to(dtype=self._strategy.config.output_dtype)
                    )

        return MaterializationReceipt(
            tensor_sha256=digest.hexdigest(),
            tensor_count=len(destination_keys),
            parameter_count=parameter_count,
            endpoint_exact=endpoint_exact,
        )
