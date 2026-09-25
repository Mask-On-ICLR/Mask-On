"""Read and write sharded model checkpoints."""

from contextlib import ExitStack
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import atomic_json, file_sha, read_json


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.stack = ExitStack()
        self.handles = {}
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.map = read_json(index)["weight_map"]
        else:
            with safe_open(self.root / "model.safetensors", framework="pt", device="cpu") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self.keys = frozenset(self.map)

    def __enter__(self):
        for name in sorted(set(self.map.values())):
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("Unsafe checkpoint shard path")
            self.handles[name] = self.stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
        return self

    def __exit__(self, *_):
        self.stack.close()

    def tensor(self, key):
        return self.handles[self.map[key]].get_tensor(key)

    def shape(self, key):
        return tuple(self.handles[self.map[key]].get_slice(key).get_shape())


def identity(root):
    root = Path(root)
    allowed = (".safetensors", ".json", ".py", ".model", ".jinja", ".txt")
    return {
        p.name: file_sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.suffix in allowed
    }


def verify_candidate(checkpoint, model, mode="diffusion"):
    from .artifacts import verify_done
    from .download import catalog

    checkpoint = Path(checkpoint)
    receipt = verify_done(checkpoint.parent)
    manifest = read_json(checkpoint.parent / "manifest.json")
    if checkpoint.name != "checkpoint" or not any(
        k.startswith("checkpoint/") for k in receipt["files"]
    ):
        raise ValueError("Checkpoint is not covered by the preparation receipt")
    if manifest.get("model") != model or manifest.get("checkpoints") != catalog()["models"][model]:
        raise ValueError("Prepared checkpoint model/anchor identity mismatch")
    if manifest.get("mode", "diffusion") != mode:
        raise ValueError("Prepared mode mismatch; use ar-view for causal merged evaluation")
    return manifest


def materialize(source, template, destination, *, shard_bytes=2_000_000_000):
    """Write one reconstruction, keeping native architecture/tokenizer/config files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tensors, mapping, total, counter = {}, {}, 0, 0

    def flush():
        nonlocal tensors, counter
        if not tensors:
            return
        name = f"model-{counter:05d}.safetensors"
        save_file(tensors, str(destination / name), metadata={"format": "pt"})
        mapping.update({k: name for k in tensors})
        tensors = {}
        counter += 1

    current = 0
    for key in sorted(source.keys):
        tensor = source.tensor(key).detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite reconstructed tensor: " + key)
        size = tensor.numel() * tensor.element_size()
        if current + size > shard_bytes:
            flush()
            current = 0
        tensors[key] = tensor.clone()
        current += size
        total += size
    flush()
    for file in Path(template).iterdir():
        if (
            file.is_file()
            and file.suffix in (".json", ".py", ".model", ".jinja", ".txt")
            and file.name != "model.safetensors.index.json"
        ):
            shutil.copy2(file, destination / file.name)
    atomic_json(
        destination / "model.safetensors.index.json",
        {"metadata": {"total_size": total}, "weight_map": mapping},
    )
    return identity(destination)


def ar_view(model, checkpoint, output):
    """Build a causal model configuration with aligned candidate weights.

    Vocabulary slicing uses the shared prefix of the AR and diffusion token ID maps.
    """
    from transformers import AutoTokenizer
    from .download import snapshot, catalog
    from .artifacts import content_sha, finish, run_directory, runtime, verify_done

    checkpoint = Path(checkpoint)
    verify_candidate(checkpoint, model)
    anchor_path = snapshot(model, "ar")
    ar_tokenizer = AutoTokenizer.from_pretrained(anchor_path, trust_remote_code=True)
    diff_tokenizer = AutoTokenizer.from_pretrained(
        snapshot(model, "diffusion"), trust_remote_code=True
    )
    if any(
        diff_tokenizer.get_vocab().get(token) != index
        for token, index in ar_tokenizer.get_vocab().items()
    ):
        raise ValueError("AR/diffusion token ID maps are not a common-prefix vocabulary")
    contract = dict(
        model=model,
        mode="ar",
        candidate=identity(checkpoint),
        anchor=identity(anchor_path),
        checkpoints=catalog()["models"][model],
        runtime=runtime(),
    )
    with run_directory(output, contract) as root:
        if (root / "DONE.json").exists():
            return verify_done(root)
        needed = sum(p.stat().st_size for p in checkpoint.glob("*.safetensors"))
        if shutil.disk_usage(root).free < needed + 2 * 1024**3:
            raise OSError("Insufficient space for native AR checkpoint view")
        with Checkpoint(checkpoint) as merged, Checkpoint(anchor_path) as anchor:

            class View:
                keys = anchor.keys

                def tensor(self, key):
                    if key not in merged.keys:
                        raise ValueError("Missing AR native tensor: " + key)
                    value = merged.tensor(key)
                    shape = anchor.shape(key)
                    if tuple(value.shape) == shape:
                        return value
                    from .prepare import is_vocab

                    if (
                        is_vocab(key)
                        and value.ndim == 2
                        and value.shape[1] == shape[1]
                        and value.shape[0] >= shape[0]
                    ):
                        return value[: shape[0]]
                    raise ValueError("Unsupported native AR tensor mapping: " + key)

            exported = materialize(View(), anchor_path, root / "checkpoint")
        atomic_json(
            root / "receipt.json", dict(files=exported, contract_sha256=content_sha(contract))
        )
        finish(root, ["manifest.json", "receipt.json"] + ["checkpoint/" + k for k in exported])
        return read_json(root / "receipt.json")
