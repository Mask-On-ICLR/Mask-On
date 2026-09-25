import torch

def exact_output_terms(x, delta, code):
    """Per-row E[u*v], E[u^2], E[v^2], including input cross terms."""
    u, v = x.float() @ code.float().T, x.float() @ delta.float().T
    return torch.stack(
        [
            (u.double() * v.double()).mean(0),
            u.double().square().mean(0),
            v.double().square().mean(0),
        ]
    ).cpu()


def positive_output_scale(terms):
    a, b, _ = terms
    return torch.where(b > 0, (a / b.clamp_min(1e-30)).clamp_min(0), torch.zeros_like(a))


def terms_error(terms, scale):
    a, b, c = terms
    return (c - 2 * scale * a + scale.square() * b).sum().clamp_min(0)
