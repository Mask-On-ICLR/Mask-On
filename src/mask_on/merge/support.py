import torch


def _validate(delta, second_moment):
    if delta.ndim != 2 or second_moment.ndim != 1 or second_moment.numel() != delta.shape[1]:
        raise ValueError('Input second moment shape mismatch')
    if not delta.is_floating_point() or not second_moment.is_floating_point():
        raise ValueError('Floating-point inputs required')
    if not torch.isfinite(delta).all() or not torch.isfinite(second_moment).all() or (second_moment<0).any():
        raise ValueError('Finite nonnegative input moments required')

def signed_top_support(
    delta: torch.Tensor,
    discard_ratio: float,
    *,
    importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select positive and negative coordinates by separate magnitude quotas.

    Optional input-channel importance weights multiply absolute delta values.
    Coordinates tied at the cutoff are retained.
    """

    if not 0.0 <= discard_ratio < 1.0:
        raise ValueError("discard_ratio must lie in [0, 1)")
    if delta.ndim != 2:
        raise ValueError("signed support requires a matrix task vector")
    if importance is None:
        score = delta.abs()
    else:
        if importance.ndim != 1 or importance.numel() != delta.shape[1]:
            raise ValueError("importance does not match the task-vector input width")
        score = delta.abs() * importance.to(delta).reshape(1, -1)
    flat_delta = delta.reshape(-1)
    flat_score = score.reshape(-1)
    support = torch.zeros_like(flat_delta, dtype=torch.bool)
    for positive in (True, False):
        partition = flat_delta > 0 if positive else flat_delta < 0
        count = int(partition.sum().item())
        if count == 0:
            continue
        keep = int(count * (1.0 - discard_ratio))
        if keep == 0:
            # Retain the sign partition when its integer quota is zero.
            support[partition] = True
            continue
        values = flat_score[partition]
        threshold = torch.kthvalue(values, values.numel() - keep + 1).values
        support |= partition & (flat_score >= threshold)
    return support.reshape(delta.shape)


def closed_form_row_scale(
    delta: torch.Tensor,
    support: torch.Tensor,
    *,
    second_moment: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a signed least-squares amplitude for each output row.

    Optional input moments apply diagonal activation weighting to the squared error.
    """

    if delta.shape != support.shape or support.dtype != torch.bool:
        raise ValueError("support must be a boolean tensor aligned with delta")
    sign = torch.where(delta > 0, 1.0, -1.0).to(delta)
    if second_moment is None:
        weight = torch.ones(delta.shape[1], dtype=delta.dtype, device=delta.device)
    else:
        _validate(delta, second_moment)
        weight = second_moment.to(delta)
    selected = support.to(delta)
    numerator = (weight.reshape(1, -1) * selected * sign * delta).sum(dim=1)
    denominator = (weight.reshape(1, -1) * selected).sum(dim=1)
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))
