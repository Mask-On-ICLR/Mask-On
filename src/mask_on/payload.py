"""Reconstruct weights from an AR anchor and compact ternary payload."""

from pathlib import Path
import torch
from safetensors.torch import load_file
from .artifacts import read_json, verify_done
from .merge.delta_compression import unpack_codes
from .checkpoints import identity


class MaskOnSource:
    def __init__(self, anchor, preparation):
        root = Path(preparation)
        verify_done(root)
        self.root = root / "payload"
        self.anchor = anchor
        self.index = read_json(self.root / "index.json")
        if self.index.get("anchor_files") != identity(anchor.root):
            raise ValueError("Compact payload anchor file identity mismatch")
        self.keys = frozenset(self.index["entries"])

    def tensor(self, key):
        row = self.index["entries"][key]
        values = load_file(str(self.root / row["file"]))
        if row["kind"] == "exact_native_exception":
            return values["exact"]
        shape = tuple(row["shape"])
        code = unpack_codes(values["code"], shape, 2).float() - 1
        scale = values["scale"]
        delta = (code.reshape(shape[0], -1) if len(shape) > 1 else code.reshape(1, -1)) * scale
        return (self.anchor.tensor(key).float() + delta.reshape(shape)).to(
            getattr(torch, row["dtype"].split(".")[-1])
        )
