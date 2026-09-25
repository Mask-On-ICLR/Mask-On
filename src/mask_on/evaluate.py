"""Atomic per-task generation, deferred sandbox scoring and tabular results."""

import csv
from pathlib import Path
import signal

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
from .download import catalog, records, snapshot
from .evaluation.common import score_record


def aggregate(rows):
    scored = [r for r in rows if r.get("score") is not None]
    seconds = sum(r["generation"]["seconds"] for r in rows)
    tokens = sum(r["generation"]["tokens"] for r in rows)
    return dict(
        generated=len(rows),
        scored=len(scored),
        correct=sum(r["score"] for r in scored),
        invalid=sum(r.get("invalid", False) for r in scored),
        accuracy=100 * sum(r["score"] for r in scored) / len(scored) if scored else None,
        status="DONE" if len(scored) == len(rows) else "SCORING",
        seconds=seconds,
        tokens=tokens,
        tps=tokens / seconds if seconds else None,
        nfe=sum(r["generation"]["nfe"] for r in rows)
        if all(r["generation"]["nfe"] is not None for r in rows)
        else None,
        timing_scope="synchronized_generation_including_prefill_not_isolation_qualified",
    )


def evaluate(
    model,
    task,
    output,
    *,
    checkpoint=None,
    mode="diffusion",
    backend="native",
    limit=None,
    device="cuda:0",
    seed=20260909,
):
    import torch
    from .models import Generator

    items = records(task)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        items = items[:limit]
    candidate = Path(checkpoint) if checkpoint else snapshot(model, mode)
    if checkpoint:
        verify_candidate(candidate, model, mode)
    contract = dict(
        model=model,
        task=task,
        mode=mode,
        backend=backend,
        checkpoints=catalog()["models"][model],
        candidate_files=identity(candidate),
        items_sha256=content_sha(items),
        item_ids=[i["id"] for i in items],
        protocol=catalog()["evaluation"],
        limit=limit,
        seed=seed,
        runtime=runtime(),
    )
    with run_directory(output, contract) as root:
        if (root / "DONE.json").exists():
            return verify_done(root)
        paused = [False]
        previous = signal.signal(signal.SIGTERM, lambda *_: paused.__setitem__(0, True))
        try:
            engine = None
            rows = []
            for i, item in enumerate(items):
                path = root / "items" / f"{i:05d}.json"
                if path.exists():
                    row = read_json(path)
                    if row["item_sha256"] != content_sha(item):
                        raise ValueError("Saved item changed")
                else:
                    if paused[0] or (root / "PAUSE").exists():
                        atomic_json(root / "PAUSED.json", dict(completed=len(rows)))
                        return dict(status="PAUSED", completed=len(rows))
                    if engine is None:
                        engine = Generator(model, mode, candidate, backend=backend, device=device)
                    ids = engine.encode(item["messages"])
                    torch.manual_seed(seed + i)
                    generation = engine.generate(ids, catalog()["evaluation"]["max_new_tokens"])
                    metrics = (
                        dict(score=None, scoring_status="pending_cpu")
                        if task in ("mbpp", "ifeval")
                        else score_record(
                            task,
                            item,
                            generation["text"],
                            choice_parser=catalog()["evaluation"]["choice_parser"]
                            if task == "arc_challenge"
                            else None,
                        )
                    )
                    if task == "gsm8k":
                        metrics["invalid"] = metrics["prediction"] == "NULL"
                    row = dict(
                        item=item,
                        item_sha256=content_sha(item),
                        prompt_ids=ids,
                        generation=generation,
                        **metrics,
                    )
                    atomic_json(path, row)
                rows.append(row)
                atomic_json(root / "progress.json", dict(completed=len(rows), total=len(items)))
            atomic_json(root / "generation_metrics.json", aggregate(rows))
            finish(
                root,
                ["manifest.json", "generation_metrics.json"]
                + [f"items/{i:05d}.json" for i in range(len(rows))],
            )
            return aggregate(rows)
        finally:
            signal.signal(signal.SIGTERM, previous)


def score(root):
    root = Path(root)
    verify_done(root)
    manifest = read_json(root / "manifest.json")
    task = manifest["task"]
    rows = [read_json(root / "items" / f"{i:05d}.json") for i in range(len(manifest["item_ids"]))]
    scoring_identity = dict(
        source_generation_sha256=content_sha(read_json(root / "DONE.json")), runtime=runtime()
    )
    scorer = None
    if task == "mbpp":
        from .evaluation.mbpp import BubblewrapPythonSandbox, MbppPassAtOneScorer, DownstreamExample

        sandbox = BubblewrapPythonSandbox(timeout_seconds=6, cpu_seconds=3)
        probe = sandbox.run("x=1", [], ["assert x == 1"])
        if not probe.passed:
            raise RuntimeError("MBPP sandbox preflight failed: " + probe.stderr)
        scorer = MbppPassAtOneScorer(sandbox)
    elif task == "ifeval":
        from .evaluation.ifeval import score_one, correction_receipt

        scoring_identity["scorer"] = correction_receipt()
    with run_directory(root / "scoring", scoring_identity) as scoring:
        if (scoring / "DONE.json").exists():
            verify_done(scoring)
            return read_json(scoring / "metrics.json")
        output = []
        for i, row in enumerate(rows):
            path = scoring / "items" / f"{i:05d}.json"
            if path.exists():
                result = read_json(path)
            else:
                if task == "mbpp":
                    record = row["item"]["record"]
                    result = scorer.score(
                        row["generation"]["text"],
                        DownstreamExample(
                            row["item"]["id"],
                            "mbpp",
                            "",
                            "",
                            dict(
                                test_imports=record.get("test_imports", []),
                                test_list=record["test_list"],
                            ),
                        ),
                    )
                elif task == "ifeval":
                    result = score_one(row["item"]["record"], row["generation"]["text"])
                else:
                    result = {k: row[k] for k in ("score", "invalid") if k in row}
                result["generation_sha256"] = content_sha(row)
                atomic_json(path, result)
            if result["generation_sha256"] != content_sha(row):
                raise ValueError("Score belongs to another generation")
            output.append(dict(row, **result))
        result = aggregate(output)
        atomic_json(scoring / "metrics.json", result)
        finish(
            scoring,
            ["manifest.json", "metrics.json"] + [f"items/{i:05d}.json" for i in range(len(rows))],
        )
    return result


def results(roots, csv_path=None):
    rows = []
    for root in map(Path, roots):
        verify_done(root)
        manifest = read_json(root / "manifest.json")
        path = root / "scoring" / "metrics.json"
        if path.exists():
            verify_done(root / "scoring")
            scoring_contract = read_json(root / "scoring/manifest.json")
            if scoring_contract["source_generation_sha256"] != content_sha(
                read_json(root / "DONE.json")
            ):
                raise ValueError("Scoring source-generation identity mismatch")
        else:
            path = root / "generation_metrics.json"
        rows.append(
            dict(
                model=manifest["model"],
                mode=manifest["mode"],
                task=manifest["task"],
                output=str(root),
                **read_json(path),
            )
        )
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model", "task"])
            writer.writeheader()
            writer.writerows(rows)
    return rows
