"""Compute streaming TIES merges from two task vectors.

Two histogram passes over nonnegative FP32 magnitude bit patterns recover
the global trimming threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from mask_on.merge.linear import MergeError, TensorSource


@dataclass(frozen=True, slots=True)
class TiesThreshold:
    """Global magnitude cutoffs and retained counts for a retention fraction."""

    retention_fraction: float
    ar_threshold: float
    diffusion_threshold: float
    floating_parameter_count: int
    ar_retained_at_threshold: int
    diffusion_retained_at_threshold: int


class GlobalTiesThresholdAuditor:
    """Compute whole-vector magnitude cutoffs with two streaming histogram passes."""

    _BUCKETS = 1 << 16

    def __init__(self, *, retention_fractions: Sequence[float], chunk_elements: int) -> None:
        fractions = tuple(sorted(set(float(value) for value in retention_fractions)))
        if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
            raise MergeError("TIES retention fractions must lie strictly inside (0, 1)")
        if chunk_elements < 1:
            raise MergeError("TIES threshold chunk size must be positive")
        self._fractions = fractions
        self._chunk_elements = chunk_elements

    def audit(
        self,
        base_source: TensorSource,
        ar_source: TensorSource,
        diffusion_source: TensorSource,
    ) -> dict[float, TiesThreshold]:
        keys = base_source.keys
        if keys != ar_source.keys or keys != diffusion_source.keys:
            raise MergeError("TIES threshold sources must align exactly")
        ar_high = torch.zeros(self._BUCKETS, dtype=torch.int64)
        diffusion_high = torch.zeros_like(ar_high)
        floating_count = 0
        for _, ar_bits, diffusion_bits in self._delta_bit_chunks(
            keys, base_source, ar_source, diffusion_source
        ):
            ar_high += torch.bincount(ar_bits >> 16, minlength=self._BUCKETS)
            diffusion_high += torch.bincount(diffusion_bits >> 16, minlength=self._BUCKETS)
            floating_count += ar_bits.numel()
        if floating_count < 1:
            raise MergeError("TIES found no floating parameters")

        ar_targets = self._target_buckets(ar_high, floating_count)
        diffusion_targets = self._target_buckets(diffusion_high, floating_count)
        ar_low = {
            bucket: torch.zeros(self._BUCKETS, dtype=torch.int64)
            for bucket in set(ar_targets.values())
        }
        diffusion_low = {
            bucket: torch.zeros(self._BUCKETS, dtype=torch.int64)
            for bucket in set(diffusion_targets.values())
        }
        for _, ar_bits, diffusion_bits in self._delta_bit_chunks(
            keys, base_source, ar_source, diffusion_source
        ):
            self._accumulate_low_bits(ar_bits, ar_low)
            self._accumulate_low_bits(diffusion_bits, diffusion_low)

        results: dict[float, TiesThreshold] = {}
        for fraction in self._fractions:
            ar_bucket = ar_targets[fraction]
            diffusion_bucket = diffusion_targets[fraction]
            ar_rank = self._within_bucket_rank(ar_high, floating_count, fraction, ar_bucket)
            diffusion_rank = self._within_bucket_rank(
                diffusion_high, floating_count, fraction, diffusion_bucket
            )
            ar_low_bits = self._ranked_bucket(ar_low[ar_bucket], ar_rank)
            diffusion_low_bits = self._ranked_bucket(
                diffusion_low[diffusion_bucket], diffusion_rank
            )
            ar_bits = (ar_bucket << 16) | ar_low_bits
            diffusion_bits = (diffusion_bucket << 16) | diffusion_low_bits
            results[fraction] = TiesThreshold(
                retention_fraction=fraction,
                ar_threshold=self._bits_to_float(ar_bits),
                diffusion_threshold=self._bits_to_float(diffusion_bits),
                floating_parameter_count=floating_count,
                ar_retained_at_threshold=self._retained_count(
                    ar_high, ar_low[ar_bucket], ar_bucket, ar_low_bits
                ),
                diffusion_retained_at_threshold=self._retained_count(
                    diffusion_high,
                    diffusion_low[diffusion_bucket],
                    diffusion_bucket,
                    diffusion_low_bits,
                ),
            )
        return results

    def _delta_bit_chunks(
        self,
        keys: frozenset[str],
        base_source: TensorSource,
        ar_source: TensorSource,
        diffusion_source: TensorSource,
    ):
        for key in sorted(keys):
            base = base_source.tensor(key)
            ar = ar_source.tensor(key)
            diffusion = diffusion_source.tensor(key)
            if not (base.shape == ar.shape == diffusion.shape):
                raise MergeError(f"TIES threshold shape mismatch for {key}")
            if not base.is_floating_point():
                if not torch.equal(base, ar) or not torch.equal(base, diffusion):
                    raise MergeError("non-floating TIES tensors must agree")
                continue
            flat_base = base.reshape(-1)
            flat_ar = ar.reshape(-1)
            flat_diffusion = diffusion.reshape(-1)
            for start in range(0, flat_base.numel(), self._chunk_elements):
                stop = min(start + self._chunk_elements, flat_base.numel())
                base_chunk = flat_base[start:stop].float()
                ar_bits = (
                    (flat_ar[start:stop].float() - base_chunk)
                    .abs()
                    .contiguous()
                    .view(torch.int32)
                    .to(torch.int64)
                )
                diffusion_bits = (
                    (flat_diffusion[start:stop].float() - base_chunk)
                    .abs()
                    .contiguous()
                    .view(torch.int32)
                    .to(torch.int64)
                )
                if (ar_bits < 0).any() or (diffusion_bits < 0).any():
                    raise MergeError("TIES delta magnitudes must be finite non-negative float32")
                yield key, ar_bits, diffusion_bits

    def _target_buckets(self, high_histogram: torch.Tensor, total: int) -> dict[float, int]:
        cumulative = torch.cumsum(high_histogram, dim=0)
        return {
            fraction: int(
                torch.searchsorted(cumulative, torch.tensor(self._kth_rank(total, fraction))).item()
            )
            for fraction in self._fractions
        }

    @staticmethod
    def _kth_rank(total: int, retention_fraction: float) -> int:
        # Use the one-indexed kthvalue rank d - int(d * K).
        return total - int(total * retention_fraction)

    def _within_bucket_rank(
        self,
        high_histogram: torch.Tensor,
        total: int,
        fraction: float,
        bucket: int,
    ) -> int:
        before = int(high_histogram[:bucket].sum().item())
        return self._kth_rank(total, fraction) - before

    @staticmethod
    def _accumulate_low_bits(bits: torch.Tensor, histograms: Mapping[int, torch.Tensor]) -> None:
        high = bits >> 16
        for bucket, histogram in histograms.items():
            selected = bits[high == bucket] & 0xFFFF
            if selected.numel():
                histogram += torch.bincount(selected, minlength=1 << 16)

    @staticmethod
    def _ranked_bucket(histogram: torch.Tensor, rank: int) -> int:
        if rank < 1 or rank > int(histogram.sum().item()):
            raise MergeError("TIES within-bucket rank is invalid")
        return int(torch.searchsorted(torch.cumsum(histogram, dim=0), torch.tensor(rank)).item())

    @staticmethod
    def _retained_count(
        high_histogram: torch.Tensor,
        low_histogram: torch.Tensor,
        high_bucket: int,
        low_bucket: int,
    ) -> int:
        return int(
            high_histogram[high_bucket + 1 :].sum().item() + low_histogram[low_bucket:].sum().item()
        )

    @staticmethod
    def _bits_to_float(bits: int) -> float:
        return float(torch.tensor(bits, dtype=torch.int32).view(torch.float32).item())


@dataclass(frozen=True, slots=True)
class TiesRecipe:
    """Retention fraction, merge coefficient, and dtype for a TIES merge."""

    name: str
    ar_weight: float
    diffusion_weight: float
    retention_fraction: float
    merge_scale: float
    variant: str = "official_mass_disjoint_mean"

    def validate(self) -> None:
        values = (
            self.ar_weight,
            self.diffusion_weight,
            self.retention_fraction,
            self.merge_scale,
        )
        if not self.name or not all(math.isfinite(value) and value > 0 for value in values):
            raise MergeError("TIES recipe values must be finite and positive")
        if not 0.0 < self.retention_fraction < 1.0:
            raise MergeError("TIES retention fraction must lie strictly inside (0, 1)")
        if self.variant not in {
            "official_mass_disjoint_mean",
            "positive_weight_adapted_mass_disjoint_mean",
        }:
            raise MergeError("unknown TIES recipe variant")
        if self.variant == "official_mass_disjoint_mean" and (
            self.ar_weight != 1.0 or self.diffusion_weight != 1.0
        ):
            raise MergeError("official TIES control requires unit input-vector weights")

    def identity(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TiesSignResolution:
    """Majority-sign value used for coordinates with a zero summed sign."""

    recipe_sign_key: str
    majority_sign: int
    positive_elected: int
    negative_elected: int
    zero_elected: int


class TiesSignAuditor:
    """Compute the global majority-sign fallback before tensor materialization."""

    def __init__(
        self,
        *,
        recipes: Sequence[TiesRecipe],
        thresholds: Mapping[float, TiesThreshold],
        chunk_elements: int,
    ) -> None:
        self._recipes = tuple(recipes)
        self._thresholds = thresholds
        self._chunk_elements = chunk_elements
        for recipe in self._recipes:
            recipe.validate()

    def audit(
        self,
        base_source: TensorSource,
        ar_source: TensorSource,
        diffusion_source: TensorSource,
    ) -> dict[str, TiesSignResolution]:
        counters = {self.sign_key(recipe): [0, 0, 0] for recipe in self._recipes}
        unique_recipes = {self.sign_key(recipe): recipe for recipe in self._recipes}
        for key in sorted(base_source.keys):
            base = base_source.tensor(key)
            ar = ar_source.tensor(key)
            diffusion = diffusion_source.tensor(key)
            if not base.is_floating_point():
                continue
            flat_base = base.reshape(-1)
            flat_ar = ar.reshape(-1)
            flat_diffusion = diffusion.reshape(-1)
            for start in range(0, flat_base.numel(), self._chunk_elements):
                stop = min(start + self._chunk_elements, flat_base.numel())
                base_chunk = flat_base[start:stop].float()
                ar_delta = flat_ar[start:stop].float() - base_chunk
                diffusion_delta = flat_diffusion[start:stop].float() - base_chunk
                for sign_key, recipe in unique_recipes.items():
                    threshold = self._thresholds[recipe.retention_fraction]
                    ar_trimmed = torch.where(
                        ar_delta.abs() >= threshold.ar_threshold,
                        recipe.ar_weight * ar_delta,
                        0.0,
                    )
                    diffusion_trimmed = torch.where(
                        diffusion_delta.abs() >= threshold.diffusion_threshold,
                        recipe.diffusion_weight * diffusion_delta,
                        0.0,
                    )
                    signs = torch.sign(ar_trimmed + diffusion_trimmed)
                    counters[sign_key][0] += int((signs > 0).sum().item())
                    counters[sign_key][1] += int((signs < 0).sum().item())
                    counters[sign_key][2] += int((signs == 0).sum().item())
        return {
            sign_key: TiesSignResolution(
                recipe_sign_key=sign_key,
                majority_sign=(1 if positive > negative else -1 if negative > positive else 0),
                positive_elected=positive,
                negative_elected=negative,
                zero_elected=zero,
            )
            for sign_key, (positive, negative, zero) in counters.items()
        }

    @staticmethod
    def sign_key(recipe: TiesRecipe) -> str:
        return json.dumps(
            {
                "ar_weight": recipe.ar_weight,
                "diffusion_weight": recipe.diffusion_weight,
                "retention_fraction": recipe.retention_fraction,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class TiesReceipt:
    """Hashes, arithmetic settings, and coordinate counts for a TIES merge."""

    tensor_sha256: str
    tensor_count: int
    parameter_count: int
    recipe: dict[str, object]
    threshold: dict[str, object]
    sign_resolution: dict[str, object]
    both_selected: int
    ar_only_selected: int
    diffusion_only_selected: int
    neither_selected: int


class TiesMaterializer:
    """Materialize a base checkpoint plus a streaming TIES task vector."""

    def __init__(
        self,
        recipe: TiesRecipe,
        *,
        threshold: TiesThreshold,
        sign_resolution: TiesSignResolution,
        chunk_elements: int,
    ) -> None:
        recipe.validate()
        if threshold.retention_fraction != recipe.retention_fraction:
            raise MergeError("TIES recipe/threshold retention mismatch")
        if sign_resolution.recipe_sign_key != TiesSignAuditor.sign_key(recipe):
            raise MergeError("TIES recipe/sign-resolution mismatch")
        self._recipe = recipe
        self._threshold = threshold
        self._sign = sign_resolution
        self._chunk_elements = chunk_elements

    def materialize(
        self,
        destination: Mapping[str, torch.Tensor],
        base_source: TensorSource,
        ar_source: TensorSource,
        diffusion_source: TensorSource,
    ) -> TiesReceipt:
        keys = frozenset(destination)
        if keys != base_source.keys or keys != ar_source.keys or keys != diffusion_source.keys:
            raise MergeError("TIES state dictionaries must align exactly")
        digest = hashlib.sha256(
            json.dumps(self._recipe.identity(), sort_keys=True).encode() + b"\0"
        )
        selected_counts = [0, 0, 0, 0]
        parameters = 0
        with torch.no_grad():
            for key in sorted(keys):
                base = base_source.tensor(key)
                ar = ar_source.tensor(key)
                diffusion = diffusion_source.tensor(key)
                target = destination[key]
                if not (base.shape == ar.shape == diffusion.shape == target.shape):
                    raise MergeError(f"shape mismatch for TIES tensor {key}")
                if not base.is_floating_point():
                    if not torch.equal(base, ar) or not torch.equal(base, diffusion):
                        raise MergeError("non-floating TIES tensors must agree")
                    target.copy_(base.to(device=target.device, dtype=target.dtype))
                    parameters += base.numel()
                    continue
                digest.update(
                    json.dumps(
                        [key, str(target.dtype), list(target.shape)], separators=(",", ":")
                    ).encode()
                    + b"\0"
                )
                flat_base = base.reshape(-1)
                flat_ar = ar.reshape(-1)
                flat_diffusion = diffusion.reshape(-1)
                flat_target = target.reshape(-1)
                for start in range(0, flat_base.numel(), self._chunk_elements):
                    stop = min(start + self._chunk_elements, flat_base.numel())
                    base_chunk = flat_base[start:stop].float()
                    ar_delta = flat_ar[start:stop].float() - base_chunk
                    diffusion_delta = flat_diffusion[start:stop].float() - base_chunk
                    ar_trimmed = torch.where(
                        ar_delta.abs() >= self._threshold.ar_threshold,
                        self._recipe.ar_weight * ar_delta,
                        0.0,
                    )
                    diffusion_trimmed = torch.where(
                        diffusion_delta.abs() >= self._threshold.diffusion_threshold,
                        self._recipe.diffusion_weight * diffusion_delta,
                        0.0,
                    )
                    elected = torch.sign(ar_trimmed + diffusion_trimmed)
                    if self._sign.majority_sign:
                        elected[elected == 0] = self._sign.majority_sign
                    ar_selected = torch.where(elected > 0, ar_trimmed > 0, ar_trimmed < 0)
                    diffusion_selected = torch.where(
                        elected > 0, diffusion_trimmed > 0, diffusion_trimmed < 0
                    )
                    count = ar_selected.to(torch.int8) + diffusion_selected.to(torch.int8)
                    merged_delta = (
                        ar_trimmed * ar_selected + diffusion_trimmed * diffusion_selected
                    ) / count.clamp(min=1)
                    merged = (base_chunk + self._recipe.merge_scale * merged_delta).to(
                        torch.bfloat16
                    )
                    flat_target[start:stop].copy_(merged.to(device=target.device))
                    digest.update(merged.contiguous().view(torch.uint8).numpy())
                    selected_counts[0] += int((count == 2).sum().item())
                    selected_counts[1] += int((ar_selected & ~diffusion_selected).sum().item())
                    selected_counts[2] += int((~ar_selected & diffusion_selected).sum().item())
                    selected_counts[3] += int((count == 0).sum().item())
                parameters += base.numel()
        return TiesReceipt(
            tensor_sha256=digest.hexdigest(),
            tensor_count=len(keys),
            parameter_count=parameters,
            recipe=self._recipe.identity(),
            threshold=asdict(self._threshold),
            sign_resolution=asdict(self._sign),
            both_selected=selected_counts[0],
            ar_only_selected=selected_counts[1],
            diffusion_only_selected=selected_counts[2],
            neither_selected=selected_counts[3],
        )
