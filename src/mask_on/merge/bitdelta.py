"""BitDelta sign matrices and trainable per-matrix scales (arXiv:2402.10193v3).

Default fitting uses Adam at 1e-4 for 200 updates with an effective batch of four.
"""

import types
from dataclasses import dataclass

import torch
from torch.nn.utils import parametrize

from .delta_compression import pack_codes, unpack_codes


@dataclass(frozen=True)
class BitDeltaConfig:
    steps: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    max_length: int = 128


PAPER_CONFIG = BitDeltaConfig()


def enable_native_moe_scale_gradients(model, family):
    """Enable gradients through LLaDA's eval-mode MoE forward during scale fitting.

    Unwrap the no_grad decorator while retaining routing and expert computation.
    """
    if family != "llada2":
        return []
    changed = []
    for name, module in model.named_modules():
        if not hasattr(module, "moe_infer"):
            continue
        bound = module.moe_infer
        wrapped = getattr(bound, "__func__", None)
        raw = getattr(wrapped, "__wrapped__", None)
        if raw is None or getattr(raw, "__name__", None) != "moe_infer":
            raise ValueError("unexpected native MoE decorator; cannot enable gradients")
        module.moe_infer = types.MethodType(raw, module)
        changed.append(name)
    if not changed:
        raise ValueError("LLaDA native MoE inference path absent")
    return changed


class BinaryWeight(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        if delta.ndim != 2 or not torch.isfinite(delta).all():
            raise ValueError("finite matrix required")
        self.register_buffer("sign", torch.where(delta > 0, 1, -1).to(torch.int8))
        self.scale = torch.nn.Parameter(delta.float().abs().mean().reshape(()))

    def forward(self, base):
        return (base.float() + self.scale * self.sign.float()).to(base.dtype)


def install_bitdelta(student, anchor, keys):
    """Install binary weights for selected projections and trainable scalar scales."""
    keys = tuple(keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("empty/duplicate projection inventory")
    params = dict(student.named_parameters())
    prepared = {}
    for key in keys:
        if key not in params or key not in anchor.keys or not key.endswith(".weight"):
            raise ValueError(f"unaligned projection {key}")
        weight = params[key]
        base = anchor.tensor(key)
        if base.shape != weight.shape or weight.ndim != 2:
            raise ValueError(f"projection shape mismatch {key}")
        module = student.get_submodule(key.rsplit(".", 1)[0])
        if parametrize.is_parametrized(module, "weight"):
            raise ValueError("already parametrized")
        prepared[key] = module
    student.requires_grad_(False)
    student.eval()
    installed = {}
    for key, module in prepared.items():
        base = anchor.tensor(key).to(module.weight.device)
        binary = BinaryWeight(module.weight.detach().float() - base.float())
        with torch.no_grad():
            module.weight.copy_(base)
        parametrize.register_parametrization(module, "weight", binary)
        installed[key] = binary
    return installed


def distill_scales(
    installed,
    examples,
    teacher_logits,
    student_logits,
    *,
    config=PAPER_CONFIG,
    step_callback=None,
    resume_state=None,
    coverage_policy="strict",
    coverage_audit=None,
):
    """Distill scales using shared teacher/student inputs and accumulated microbatches.

    Run 200 updates at effective batch size four and export the final scales.
    """
    if not examples or config.steps < 1 or config.batch_size < 1:
        raise ValueError("invalid fit budget")
    parameters = [m.scale for m in installed.values()]
    initial_scales = {k: m.scale.detach().clone() for k, m in installed.items()}
    if coverage_policy not in ("strict", "retain_initial_unobserved_experts"):
        raise ValueError("unknown BitDelta coverage policy")
    optimizer = torch.optim.Adam(
        parameters, lr=config.learning_rate, betas=config.betas, eps=config.eps
    )
    start_step = 0
    observed = {k: 0 for k in installed}
    if resume_state:
        if list(resume_state["keys"]) != list(installed):
            raise ValueError("resume scale inventory mismatch")
        for key, value in resume_state["scales"].items():
            installed[key].scale.data.copy_(value.to(installed[key].scale))
        optimizer.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state["step"])
        observed.update(resume_state["observed"])
        if set(resume_state["scales"]) != set(installed) or set(observed) != set(installed):
            raise ValueError("resume coverage inventory mismatch")
        if start_step < 0 or start_step > config.steps:
            raise ValueError("resume update out of range")
    trace = []
    for step in range(start_step, config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for offset in range(config.batch_size):
            example = examples[(step * config.batch_size + offset) % len(examples)]
            with torch.no_grad():
                target = teacher_logits(example).detach()
            output = student_logits(example)
            if target.shape != output.shape:
                raise ValueError("teacher/student logit shape mismatch")
            loss = (output.float() - target.to(output.device).float()).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite logit MSE")
            (loss / config.batch_size).backward()
            loss_sum += loss.item() / config.batch_size
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("nonfinite scale gradient")
        for key, module in installed.items():
            observed[key] += int(module.scale.grad is not None)
        if not any(p.grad is not None for p in parameters):
            raise ValueError("all scale gradients are absent")
        optimizer.step()
        row = dict(step=step + 1, logit_mse=loss_sum, observed=observed.copy())
        trace.append(row)
        if step_callback:
            step_callback(row, installed, optimizer)
    missing = [k for k, count in observed.items() if not count]
    if missing and coverage_policy == "strict":
        raise ValueError(f"full-fit scale coverage missing, no silent fallback: {missing[:16]}")
    for key in missing:
        if ".experts." not in key or not torch.equal(installed[key].scale, initial_scales[key]):
            raise ValueError(f"unobserved expert scale is not its original initialization: {key}")
    if any(not torch.isfinite(m.scale).all() for m in installed.values()):
        raise ValueError("nonfinite final scale")
    if coverage_audit is not None:
        coverage_audit.update(policy=coverage_policy, observed_matrices=len(installed)-len(missing),
                              total_matrices=len(installed), initial_retained=missing,
                              initial_retained_bit_exact=True, additional_updates=config.steps-start_step)
    return trace


def export_bitdelta(installed):
    metadata, tensors = {}, {}
    for i, (key, module) in enumerate(installed.items()):
        prefix = f"m{i}"
        metadata[key] = dict(method="bitdelta", prefix=prefix, shape=list(module.sign.shape))
        tensors[prefix + ".codes"] = pack_codes(module.sign > 0, 1)
        tensors[prefix + ".scale"] = module.scale.detach().cpu().float()
    return metadata, tensors


def reconstruct_bitdelta(meta, get):
    p = meta["prefix"]
    return (2 * unpack_codes(get(p + ".codes"), meta["shape"], 1) - 1) * get(p + ".scale")
