"""Download model checkpoints, datasets, and decoders at pinned revisions."""

import json
import os
from importlib.resources import files
from pathlib import Path
import subprocess

from .artifacts import atomic_json, file_sha, finish, read_json


def catalog():
    return json.loads(files("mask_on").joinpath("catalog.json").read_text())


def home():
    return Path(os.environ.get("MASK_ON_HOME", "artifacts")).expanduser().resolve()


def dataset_cache():
    # Store Arrow caches in datasets-version-specific directories.
    import datasets

    return str(home() / "cache" / f"datasets-{datasets.__version__}")


def snapshot(model, mode):
    from huggingface_hub import snapshot_download

    repo, revision = catalog()["models"][model]["anchor" if mode == "ar" else "endpoint"]
    return Path(snapshot_download(repo_id=repo, revision=revision, local_files_only=True))


def download(models, tasks, *, dry_run=False, metadata_only=False):
    plan = {
        "models": {
            m: {k: catalog()["models"][m][k] for k in ("anchor", "endpoint")} for m in models
        },
        "tasks": {t: catalog()["datasets"][t] for t in tasks},
    }
    if dry_run:
        return plan
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    from .evaluation.common import prepare_records

    for pairs in plan["models"].values():
        for repo, revision in pairs.values():
            snapshot_download(
                repo_id=repo,
                revision=revision,
                ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],
                allow_patterns=["*.py", "*.json", "*.jinja", "*.model", "*.txt"]
                if metadata_only
                else None,
            )
    for task, spec in plan["tasks"].items():
        folder = home() / "datasets" / task
        if (folder / "DONE.json").exists():
            from .artifacts import verify_done

            verify_done(folder)
            if read_json(folder / "manifest.json")["dataset"] != spec:
                raise ValueError("Existing dataset contract mismatch")
            continue
        dataset = load_dataset(
            spec["repo_id"],
            spec["subset"],
            revision=spec["revision"],
            split=spec["split"],
            cache_dir=dataset_cache(),
        )
        if len(dataset) != spec["count"]:
            raise ValueError("Dataset count mismatch: " + task)
        atomic_json(folder / "records.json", prepare_records(task, list(dataset), shots=0))
        atomic_json(
            folder / "manifest.json",
            {"dataset": spec, "shots": 0, "records_sha256": file_sha(folder / "records.json")},
        )
        finish(folder, ["manifest.json", "records.json"])
    return plan


def native_source(model, fetch=False):
    repo, revision = catalog()["native_sources"][model]
    root = home() / "sources" / model
    if fetch and not root.exists():
        subprocess.run(["git", "clone", "--no-checkout", repo, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if actual != revision or dirty:
        raise ValueError("Native engine source revision/cleanliness mismatch")
    return root


def records(task):
    from .artifacts import verify_done

    root = home() / "datasets" / task
    verify_done(root)
    return read_json(root / "records.json")
