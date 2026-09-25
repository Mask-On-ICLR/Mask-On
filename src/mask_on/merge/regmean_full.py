import torch

def regularized_pair(grams, relative_loading=0.0):
    """Add endpoint-centered diagonal loading to the two normal-equation Grams.

    The loading is eta * trace(G_A + G_D) / (2 * input_width).
    """
    import math
    if not math.isfinite(relative_loading) or relative_loading < 0:
        raise ValueError('invalid RegMean relative diagonal loading')
    matrices = [.9*g + .1*torch.diag(g.diagonal()) for g in grams]
    beta = relative_loading * float(sum(g.trace() for g in grams)) / (2 * len(grams[0]))
    if relative_loading and (not math.isfinite(beta) or beta <= 0):
        raise ValueError('positive observed Gram energy required for relative loading')
    if beta:
        matrices = [g + beta*torch.eye(len(g), device=g.device, dtype=g.dtype) for g in matrices]
    return matrices, beta


def merge_from_full_grams(ar_weight, diffusion_weight, ar_gram, diffusion_gram, *, relative_loading=0.0):
    """Solve RegMean using row-normalized full Grams and equal model weights."""
    if ar_weight.ndim != 2 or ar_weight.shape != diffusion_weight.shape:
        raise ValueError('aligned dense weights required')
    width = ar_weight.shape[1]
    matrices = []
    for gram in (ar_gram, diffusion_gram):
        if gram.shape != (width, width) or not torch.isfinite(gram).all():
            raise ValueError('invalid offline mean Gram')
        g = gram.to(device=ar_weight.device, dtype=torch.float64)
        torch.testing.assert_close(g, g.T, rtol=1e-7, atol=1e-8)
        matrices.append(g)
    (ga, gd), _ = regularized_pair(matrices, relative_loading)
    rhs = ga @ ar_weight.double().T + gd @ diffusion_weight.double().T
    out = torch.linalg.solve(ga + gd, rhs).T
    if not torch.isfinite(out).all():
        raise ValueError('nonfinite original RegMean solve')
    residual = ((ga+gd) @ out.T-rhs).norm()/rhs.norm().clamp_min(1e-12)
    if residual > 1e-8:
        raise ValueError('RegMean solve residual exceeds registered tolerance')
    return out.to(ar_weight.dtype)
