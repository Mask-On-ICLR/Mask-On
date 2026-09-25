"""Atomic JSON writes, file hashes, runtime metadata, and run-directory locking."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def content_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity():
    root = Path(__file__).parent
    return content_sha(
        {
            str(p.relative_to(root)): file_sha(p)
            for p in sorted(root.rglob("*"))
            if p.suffix in (".py", ".json")
        }
    )


def runtime():
    versions = {}
    for name in ("mask-on", "torch", "transformers", "datasets", "safetensors", "vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "packages": versions,
        "source_sha256": source_identity(),
    }


@contextmanager
def run_directory(root, contract):
    """Lock the run directory, validate its manifest, and record execution failures."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "manifest.json"
        if path.exists() and read_json(path) != contract:
            raise ValueError("Resume contract changed; choose a fresh output directory")
        if not path.exists():
            atomic_json(path, contract)
        if (root / "FAILED.json").exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            (root / "failures").mkdir(exist_ok=True)
            os.replace(root / "FAILED.json", root / "failures" / f"retry-{stamp}.json")
        try:
            yield root
        except BaseException as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            failure = {"type": type(exc).__name__, "message": str(exc)}
            atomic_json(root / "failures" / f"{stamp}.json", failure)
            atomic_json(root / "FAILED.json", failure)
            raise


def finish(root, files):
    atomic_json(Path(root) / "DONE.json", {"files": {f: file_sha(Path(root) / f) for f in files}})


def verify_done(root):
    root = Path(root)
    receipt = read_json(root / "DONE.json")
    if not isinstance(receipt.get("files"), dict) or "manifest.json" not in receipt["files"]:
        raise ValueError("Missing manifest-bound file inventory")
    for name, digest in receipt["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha(path) != digest:
            raise ValueError("Artifact hash mismatch: " + name)
    return receipt
