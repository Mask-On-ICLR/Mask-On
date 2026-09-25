"""Generate and save a diffusion draft, then review it with a native AR model.

The two models use separate dense weights and per-stage output records.
"""

from pathlib import Path
import time

from .artifacts import (
    atomic_json,
    content_sha,
    finish,
    read_json,
    run_directory,
    runtime,
    verify_done,
)
from .checkpoints import identity, verify_candidate
from .download import catalog, records
from .evaluation.gsm8k_metrics import equal, postprocess
from .evaluation.mode_cascade import parse_review

REVIEW_PROMPT = """{problem}

A student produced this solution:
{solution}

Check the reasoning and final number. If correct, reply CORRECT.
If incorrect, reply INCORRECT, explain and correct the solution step by step,
then end with a line containing #### <number>."""


def execute_item(root, index, item, draft, review):
    root = Path(root)
    dpath, rpath = root / "drafts" / f"{index:05d}.json", root / "reviews" / f"{index:05d}.json"
    resumed = dpath.exists()
    start = time.perf_counter()
    if dpath.exists():
        d = read_json(dpath)
        if d["item_sha256"] != content_sha(item):
            raise ValueError("Draft item drift")
    else:
        generated = draft(item["messages"])  # Generate from the prompt messages.
        prediction = postprocess(generated["text"])
        d = dict(
            item_sha256=content_sha(item),
            generation=generated,
            prediction=prediction,
            correct=equal(prediction, item["reference"]),
            invalid=prediction == "NULL",
        )
        atomic_json(dpath, d)
    if rpath.exists():
        r = read_json(rpath)
        if r["draft_sha256"] != content_sha(d):
            raise ValueError("Review is bound to a different draft")
        # Rebuild a missing item record from the saved review.
        atomic_json(
            root / "items" / f"{index:05d}.json",
            dict(draft_sha256=content_sha(d), review_sha256=content_sha(r)),
        )
        return dict(draft=d, review=r)
    question = item["record"]["question"]
    text = REVIEW_PROMPT.format(problem=question, solution=d["generation"]["text"])
    generated = review([{"role": "user", "content": text}])
    parsed = parse_review(generated["text"], d["prediction"], policy="review_semantic_v3")
    prediction = (
        d["prediction"]
        if parsed["verdict"] == "accept"
        else parsed["correction"]
        if parsed["verdict"] == "reject"
        else "NULL"
    )
    r = dict(
        draft_sha256=content_sha(d),
        generation=generated,
        parsed=parsed,
        prediction=prediction,
        correct=equal(prediction, item["reference"]),
        invalid=prediction == "NULL",
        composed_generation_seconds=d["generation"]["seconds"] + generated["seconds"],
        continuous_item_wall_seconds=None if resumed else time.perf_counter() - start,
        resumed_draft=resumed,
    )
    atomic_json(rpath, r)
    atomic_json(
        root / "items" / f"{index:05d}.json",
        dict(draft_sha256=content_sha(d), review_sha256=content_sha(r)),
    )
    return dict(draft=d, review=r)


def summarize(rows):
    n = len(rows)

    def stage(name):
        values = [r[name] for r in rows]
        seconds = sum(v["generation"]["seconds"] for v in values)
        tokens = sum(v["generation"]["tokens"] for v in values)
        return dict(
            count=n,
            correct=sum(v["correct"] for v in values),
            invalid=sum(v["invalid"] for v in values),
            accuracy=100 * sum(v["correct"] for v in values) / n if n else None,
            seconds=seconds,
            tokens=tokens,
            tps=tokens / seconds if seconds else None,
            nfe=sum(v["generation"]["nfe"] for v in values)
            if all(v["generation"].get("nfe") is not None for v in values)
            else None,
            peak_memory_bytes=max(
                (v["generation"].get("peak_memory_bytes") or 0 for v in values), default=0
            ),
        )

    d, r = stage("draft"), stage("review")
    return dict(
        diffusion_only=d,
        review=r,
        wrong_to_correct=sum(not x["draft"]["correct"] and x["review"]["correct"] for x in rows),
        correct_to_wrong=sum(x["draft"]["correct"] and not x["review"]["correct"] for x in rows),
        combined_generation_tps=(d["tokens"] + r["tokens"]) / (d["seconds"] + r["seconds"])
        if d["seconds"] + r["seconds"]
        else None,
        continuous_items=sum(x["review"]["continuous_item_wall_seconds"] is not None for x in rows),
        continuous_item_wall_seconds=sum(
            x["review"]["continuous_item_wall_seconds"] or 0 for x in rows
        ),
        composed_generation_seconds=d["seconds"] + r["seconds"],
        cost_scope="two_dense_accuracy_reference; fresh_AR_prefill_in_review; not_single_resident_switching",
    )


def run(checkpoint, output, threshold=0.9, limit=None, device="cuda:0"):
    from .models import Generator
    import torch

    checkpoint = Path(checkpoint)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    verify_candidate(checkpoint, "fast")
    items = records("gsm8k")[:limit]
    contract = dict(
        model="fast",
        threshold=threshold,
        checkpoints=catalog()["models"]["fast"],
        candidate_files=identity(checkpoint),
        items_sha256=content_sha(items),
        budget=1024,
        residency="two_dense_accuracy_reference",
        parser="review_semantic_v3",
        review_prompt=REVIEW_PROMPT,
        runtime=runtime(),
    )
    with run_directory(output, contract) as root:
        if (root / "DONE.json").exists():
            return verify_done(root)
        drafter = Generator("fast", checkpoint=checkpoint, threshold=threshold, device=device)
        reviewer = Generator("fast", mode="ar", device=device)

        def stage(engine):
            return lambda messages: engine.generate(engine.encode(messages), 1024)

        rows = []
        for i, item in enumerate(items):
            if (root / "PAUSE").exists():
                atomic_json(root / "PAUSED.json", dict(completed=i))
                return dict(status="PAUSED", completed=i)
            torch.manual_seed(20260909 + i)
            rows.append(execute_item(root, i, item, stage(drafter), stage(reviewer)))
            atomic_json(root / "progress.json", dict(completed=i + 1, total=len(items)))
        result = summarize(rows)
        atomic_json(root / "metrics.json", result)
        finish(
            root,
            ["manifest.json", "metrics.json"]
            + [
                f"{kind}/{i:05d}.json"
                for kind in ("drafts", "reviews", "items")
                for i in range(len(items))
            ],
        )
        return result
