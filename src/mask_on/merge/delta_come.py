"""Compress task deltas with rank allocation and grouped GPTQ (Delta-CoMe Eq. 5-7)."""

import math
from dataclasses import dataclass

import torch

from .delta_compression import pack_codes, unpack_codes


@dataclass(frozen=True)
class DeltaCoMeConfig:
    group_size: int = 128
    block_size: int = 128
    damp_percent: float = 0.01
    activation_order: bool = True
    max_length: int = 2048
    first_rank: int = 2
    second_rank: int = 32
    nominal_bits_per_delta_element: float = 1.0


PAPER_CONFIG = DeltaCoMeConfig()


def paper_rank_plan(shape, config=PAPER_CONFIG):
    """Allocate ranks satisfying 8*2 + 3*32 + 2*r <= hout*hin/(hout+hin).

    The budget counts factor codes; scales and index metadata are stored separately.
    """
    out_features, in_features = shape
    budget = (
        config.nominal_bits_per_delta_element
        * out_features
        * in_features
        / (out_features + in_features)
    )
    r2 = math.floor((budget - 8 * config.first_rank - 3 * config.second_rank) / 2)
    r2 = min(r2, min(shape) - config.first_rank - config.second_rank)
    if r2 < 0:
        raise ValueError(
            "matrix too small for the published 8+3+2 allocation; explicit policy required"
        )
    return [(8, config.first_rank), (3, config.second_rank), (2, r2)]


def _affine_parameters(weight, bits):
    zero = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    low = torch.minimum(weight.amin(1), zero)
    high = torch.maximum(weight.amax(1), zero)
    empty = (low == 0) & (high == 0)
    low = torch.where(empty, -torch.ones_like(low), low)
    high = torch.where(empty, torch.ones_like(high), high)
    scale = (high - low) / (2**bits - 1)
    return scale, torch.round(-low / scale)


@torch.no_grad()
def gptq_factor(weight, gram, bits, config=PAPER_CONFIG):
    """Quantize factor rows with asymmetric grouped GPTQ using the input Gram.

    Reject nonfinite matrices and Grams with no observed input variance.
    """
    if bits not in (2, 3, 4, 8) or weight.ndim != 2:
        raise ValueError("GPTQ factor requires a matrix and 2/3/4/8 bits")
    if config.group_size < 1 or config.block_size < 1 or config.damp_percent <= 0:
        raise ValueError("invalid GPTQ configuration")
    w = weight.detach().float().clone()
    # Compute the damped Gram solve in FP64.
    h = gram.to(device=w.device, dtype=torch.float64).clone()
    n = w.shape[1]
    if h.shape != (n, n) or not torch.isfinite(h).all() or not torch.isfinite(w).all():
        raise ValueError("nonfinite or incompatible Gram/factor")
    if not torch.allclose(h, h.T, atol=1e-4, rtol=1e-4) or (h.diag() < 0).any():
        raise ValueError("invalid Gram symmetry/diagonal")
    if not (h.diag() > 0).any():
        raise ValueError("missing activation coverage; no silent MoE fallback")
    h = (h + h.T) * 0.5
    dead = h.diag() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    order = (
        torch.argsort(h.diag(), descending=True, stable=True)
        if config.activation_order
        else torch.arange(n, device=w.device)
    )
    w, h = w[:, order], h[order][:, order]
    h.diagonal().add_(config.damp_percent * h.diag().mean())
    # Factor the Gram with the configured diagonal damping.
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
    upper = torch.linalg.cholesky(inverse, upper=True).to(w.dtype)
    q = torch.zeros_like(w)
    codes = torch.zeros_like(w, dtype=torch.uint8)
    scales, zeros = [], []
    scale = zero = None
    for start in range(0, n, config.block_size):
        end = min(start + config.block_size, n)
        local = w[:, start:end].clone()
        errors = torch.zeros_like(local)
        for j in range(end - start):
            column = start + j
            if column % config.group_size == 0:
                # Apply within-block updates before fitting the next group's scale.
                w[:, column:end] = local[:, j:]
                scale, zero = _affine_parameters(
                    w[:, column : min(column + config.group_size, n)], bits
                )
                scales.append(scale.clone())
                zeros.append(zero.clone())
            original = local[:, j]
            code = (torch.round(original / scale) + zero).clamp(0, 2**bits - 1)
            quant = scale * (code - zero)
            q[:, column], codes[:, column] = quant, code.to(torch.uint8)
            error = (original - quant) / upper[column, column]
            local[:, j:] -= error[:, None] * upper[column, column:end][None]
            errors[:, j] = error
        w[:, end:] -= errors @ upper[start:end, end:]
    inverse_order = torch.argsort(order)
    result = dict(
        shape=list(weight.shape),
        bits=bits,
        group_size=config.group_size,
        codes=pack_codes(codes, bits),
        scales=torch.stack(scales, 1).cpu(),
        zeros=torch.stack(zeros, 1).cpu(),
        order=order.cpu().to(torch.int32),
    )
    return q[:, inverse_order], result


def decode_factor(meta, get, prefix):
    codes = unpack_codes(get(prefix + ".codes"), meta["shape"], meta["bits"])
    groups = torch.arange(meta["shape"][1]) // meta["group_size"]
    value = get(prefix + ".scales")[:, groups] * (codes - get(prefix + ".zeros")[:, groups])
    return value[:, torch.argsort(get(prefix + ".order").long())]


def export_factor(result, prefix, tensors):
    for name in ("codes", "scales", "zeros", "order"):
        tensors[prefix + "." + name] = result[name]
    return {k: result[k] for k in ("shape", "bits", "group_size")}


@torch.no_grad()
def quantize_component(u, s, vt, gram, bits, config=PAPER_CONFIG):
    vhat, vp = gptq_factor(vt, gram, bits, config)
    intermediate = s[:, None] * vhat
    # Compute the Gram of Sigma Vhat^T X.
    high_precision = intermediate.double()
    ugram = (
        high_precision @ gram.to(device=intermediate.device, dtype=torch.float64) @ high_precision.T
    )
    uhat, up = gptq_factor(u, ugram, bits, config)
    return (uhat * s[None]) @ vhat, (up, s.detach().cpu().float(), vp)


@torch.no_grad()
def compress_delta_come(delta, gram, *, config=PAPER_CONFIG, plan=None, prefix="m0"):
    """Compress one projection delta into quantized singular-vector components."""
    if delta.ndim != 2 or not torch.isfinite(delta).all():
        raise ValueError("finite delta matrix required")
    plan = paper_rank_plan(delta.shape, config) if plan is None else plan
    if sum(r for _, r in plan) > min(delta.shape) or any(r < 0 for _, r in plan):
        raise ValueError("invalid singular-rank allocation")
    u, s, vt = torch.linalg.svd(delta.float(), full_matrices=False)
    components, tensors, start = [], {}, 0
    for index, (bits, rank) in enumerate(plan):
        if not rank:
            continue
        end = start + rank
        _, (up, sp, vp) = quantize_component(
            u[:, start:end], s[start:end], vt[start:end], gram, bits, config
        )
        p = f"{prefix}.c{index}"
        components.append(
            dict(
                prefix=p,
                u=export_factor(up, p + ".u", tensors),
                v=export_factor(vp, p + ".v", tensors),
                start=start,
                end=end,
            )
        )
        tensors[p + ".s"] = sp
        start = end
    return dict(
        method="delta_come",
        shape=list(delta.shape),
        components=components,
        plan=[list(x) for x in plan],
        omitted_rank=min(delta.shape) - start,
    ), tensors


def reconstruct_delta_come(meta, get):
    result = torch.zeros(meta["shape"], dtype=torch.float32)
    for c in meta["components"]:
        p = c["prefix"]
        u = decode_factor(c["u"], get, p + ".u")
        vt = decode_factor(c["v"], get, p + ".v")
        result += (u * get(p + ".s")[None]) @ vt
    return result
