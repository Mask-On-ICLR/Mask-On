"""Optimize merge coefficients and select the minimum-validation-loss state."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mask_on.evaluation.adamerging_calibration import panel_audit
from mask_on.merge.adamerging_layerwise import BestValidation, digest, file_digest

DOMAINS = ("code", "math", "stem", "chat")


def balanced_order(examples, steps, seed):
    """Return one shuffled fitting item per domain and optimizer update."""
    rng = random.Random(seed)
    pools = {
        domain: sorted(
            (
                i
                for i, e in enumerate(examples)
                if e["row"]["split"] == "fit" and e["row"]["domain"] == domain
            ),
            key=lambda i: examples[i]["row"]["prompt_sha256"],
        )
        for domain in DOMAINS
    }
    if any(len(pool) != 32 for pool in pools.values()):
        raise ValueError("each fitting domain must have 32 items")
    order = []
    for step in range(steps):
        if step % 32 == 0:
            for pool in pools.values():
                rng.shuffle(pool)
        order.append([pools[d][step % 32] for d in DOMAINS])
    return order


@torch.no_grad()
def evaluate_validation(objective, examples):
    values = []
    for example in examples:
        if example["row"]["split"] != "validation":
            raise ValueError("non-validation item in checkpoint selection")
        loss = float(objective.loss(example))
        if not math.isfinite(loss):
            raise ValueError("nonfinite validation loss")
        values.append(
            dict(
                prompt_sha256=example["row"]["prompt_sha256"],
                domain=example["row"]["domain"],
                loss=loss,
            )
        )
    if len(values) != 32:
        raise ValueError("checkpoint selection requires all 32 validation items")
    return sum(r["loss"] for r in values) / 32, values


def fit_coefficients(
    objective, examples, bank, *, mapping, binding, root, qualification, progress_callback=None
):
    """Fit merge coefficients with Adam and save the lowest-validation-loss state.

    Validate forward parity, coefficient gradients, frozen weights, and aliases
    before creating a new output directory.
    """
    panel_audit([e["row"] for e in examples])
    config = bank.config
    config.validate()
    required = (
        "native_forward_parity",
        "coefficient_gradient",
        "frozen_weights",
        "tied_alias_parity",
        "active_expert_gradient_if_applicable",
    )
    if (
        qualification.get("status") != "PASS"
        or qualification.get("binding_sha256") != digest(binding)
        or not qualification.get("evidence_sha256")
        or any(qualification.get(k) is not True for k in required)
    ):
        raise ValueError("missing independent native fitting qualification")
    if set(mapping) != set(bank.keys):
        raise ValueError("coefficient mapping mismatch")
    if hasattr(objective, "model") and any(p.requires_grad for p in objective.model.parameters()):
        raise ValueError("endpoint weights must remain frozen; only coefficients may train")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=1,
        config=asdict(config),
        binding=binding,
        qualification=qualification,
        mapping=mapping,
        trainable="merge_coefficients_only",
        full_generation=False,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    best, trace, started = BestValidation(), [], time.monotonic()
    valid = sorted(
        (e for e in examples if e["row"]["split"] == "validation"),
        key=lambda e: (e["row"]["domain"], e["row"]["prompt_sha256"]),
    )
    order = balanced_order(examples, config.steps, config.seed)
    optimizer = torch.optim.Adam(
        [bank.raw], lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=0.0
    )

    def record(row):
        row = dict(row, elapsed_seconds=time.monotonic() - started)
        trace.append(row)
        with (root / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress_callback is not None:
            progress_callback(row)

    def checkpoint(step):
        loss, items = evaluate_validation(objective, valid)
        selected = best.consider(step=step, loss=loss, bank=bank)
        record(
            dict(
                kind="validation",
                step=step,
                loss=loss,
                selected=selected,
                items=items,
                coefficient_values_sha256=digest(bank().detach().cpu().tolist()),
            )
        )

    try:
        checkpoint(0)
        for step, indices in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i in indices:
                loss = objective.loss(examples[i])
                if loss.ndim or not torch.isfinite(loss):
                    raise ValueError("invalid/nonfinite fit loss")
                (loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            grad = bank.raw.grad
            if grad is None or not torch.isfinite(grad).all():
                raise ValueError("missing/nonfinite coefficient gradient")
            if step == 1 and not torch.count_nonzero(grad):
                raise ValueError("all initial coefficient gradients are zero")
            grad_norm = float(grad.norm())
            optimizer.step()
            # Clamp effective coefficients during forward; keep the raw Adam state.
            if not torch.isfinite(bank.raw).all():
                raise ValueError("nonfinite optimizer state")
            record(
                dict(
                    kind="fit",
                    step=step,
                    loss=sum(losses) / len(losses),
                    gradient_norm=grad_norm,
                    item_ids=[examples[i]["row"]["prompt_sha256"] for i in indices],
                )
            )
            if step % config.validation_interval == 0 or step == config.steps:
                checkpoint(step)
        receipt = best.save(
            root / "selected", bank=bank, mapping=mapping, binding=binding, trace=trace
        )
        # Reload the selected coefficients and recompute validation losses.
        from mask_on.merge.adamerging_layerwise import load_coefficients

        values, _ = load_coefficients(
            root / "selected",
            expected_receipt_sha256=file_digest(root / "selected/receipt.json"),
            expected_binding=binding,
        )
        with torch.no_grad():
            bank.raw.copy_(torch.tensor([values[k] for k in bank.keys], device=bank.raw.device))
        reloaded, _ = evaluate_validation(objective, valid)
        if not math.isclose(reloaded, best.loss, abs_tol=1e-6, rel_tol=1e-5):
            raise ValueError("best coefficient reload validation mismatch")
        aggregate = dict(
            status="DONE",
            best_step=best.step,
            validation_loss=best.loss,
            reload_validation_loss=reloaded,
            optimizer_steps=config.steps,
            fit_items=128,
            validation_items=32,
            selected_receipt_sha256=file_digest(root / "selected/receipt.json"),
            coefficient_sha256=receipt["coefficient_sha256"],
            elapsed_seconds=time.monotonic() - started,
            forward_count=getattr(objective, "forward_count", None),
            training_complete_not_benchmark_result=True,
        )
        (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
        (root / "DONE").write_text("DONE\n")
        return aggregate
    except BaseException as exc:
        failure = dict(
            status="FAILED",
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        (root / "FAILED.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        # Remove the selected-state completion marker after an audit failure.
        if (root / "selected").is_dir():
            (root / "selected/FAILED.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


def audit_completed_fit(root, *, expected_aggregate_sha256):
    """Verify completion, minimum validation loss, and saved coefficient hashes."""
    from mask_on.merge.adamerging_layerwise import load_coefficients

    root = Path(root)
    if not (root / "DONE").is_file() or (root / "FAILED.json").exists():
        raise ValueError("fitting run is not successfully terminal")
    if file_digest(root / "aggregate.json") != expected_aggregate_sha256:
        raise ValueError("fit aggregate drift")
    aggregate = json.loads((root / "aggregate.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    values, receipt = load_coefficients(
        root / "selected",
        expected_receipt_sha256=aggregate["selected_receipt_sha256"],
        expected_binding=manifest["binding"],
    )
    if receipt["config"] != manifest["config"] or receipt["mapping"] != manifest["mapping"]:
        raise ValueError("fit config/mapping drift")
    trace = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    if trace != receipt["trace"]:
        raise ValueError("fitting trace drift")
    config = receipt["config"]
    steps, interval = config["steps"], config["validation_interval"]
    validation = [r for r in trace if r["kind"] == "validation"]
    expected = sorted({0, steps} | set(range(interval, steps + 1, interval)))
    if [r["step"] for r in validation] != expected:
        raise ValueError("incomplete registered validation checkpoints")
    if [r["step"] for r in trace if r["kind"] == "fit"] != list(range(1, steps + 1)):
        raise ValueError("incomplete registered fitting updates")
    identities = None
    from collections import Counter

    for row in validation:
        items = row["items"]
        ids = {i["prompt_sha256"] for i in items}
        if (
            len(items) != 32
            or len(ids) != 32
            or Counter(i["domain"] for i in items) != {d: 8 for d in DOMAINS}
            or any(not math.isfinite(i["loss"]) for i in items)
        ):
            raise ValueError("invalid validation item inventory")
        if identities is not None and ids != identities:
            raise ValueError("validation identities changed during fitting")
        identities = ids
        if not math.isclose(row["loss"], sum(i["loss"] for i in items) / 32, abs_tol=1e-9):
            raise ValueError("validation loss reduction mismatch")
    winner = min(validation, key=lambda row: (row["loss"], row["step"]))
    if (
        receipt["best_step"] != winner["step"]
        or receipt["validation_loss"] != winner["loss"]
        or aggregate["best_step"] != winner["step"]
        or aggregate["validation_loss"] != winner["loss"]
        or winner["coefficient_values_sha256"] != digest([values[k] for k in receipt["keys"]])
    ):
        raise ValueError("saved coefficients are not the observed validation winner")
    return dict(
        status="COMPLETED_FIT_AUDIT_PASS",
        best_step=winner["step"],
        validation_loss=winner["loss"],
        binding=receipt["binding"],
        selected_receipt_sha256=aggregate["selected_receipt_sha256"],
    )
