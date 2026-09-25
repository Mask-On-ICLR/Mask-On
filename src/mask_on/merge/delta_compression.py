"""Pack quantized task deltas and reconstruct dense tensors from stored payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack_codes(codes, bits):
    if bits not in (1, 2, 3, 4, 8):
        raise ValueError("unsupported bit width")
    a = codes.detach().cpu().numpy().reshape(-1)
    if np.any(a < 0) or np.any(a >= 2**bits) or np.any(a != np.floor(a)):
        raise ValueError("noninteger/out-of-range code")
    planes = (a.astype(np.uint8)[:, None] >> np.arange(bits)) & 1
    return torch.from_numpy(np.packbits(planes.reshape(-1), bitorder="little"))


def unpack_codes(packed, shape, bits, *, device="cpu"):
    count = int(np.prod(shape))
    a = np.unpackbits(packed.cpu().numpy(), bitorder="little")[: count * bits]
    if len(a) != count * bits:
        raise ValueError("short packed payload")
    values = (a.reshape(-1, bits) * (1 << np.arange(bits))).sum(1)
    return torch.as_tensor(values.reshape(shape), device=device, dtype=torch.float32)


def write_payload(root, metadata, tensors, binding):
    """Save compressed tensors, metadata, hashes, and a completion marker."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if not binding or not metadata:
        raise ValueError("binding/inventory required")
    tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("nonfinite payload")
    save_file(tensors, str(root / "payload.safetensors"))
    receipt = dict(
        schema_version=1,
        binding=binding,
        matrices=metadata,
        payload_sha256=file_sha256(root / "payload.safetensors"),
        serialized_tensor_bytes=sum(v.numel() * v.element_size() for v in tensors.values()),
        file_bytes=(root / "payload.safetensors").stat().st_size,
    )
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    sha = file_sha256(root / "receipt.json")
    (root / "DONE.json").write_text(json.dumps(dict(receipt_sha256=sha)) + "\n")
    return sha


class CompressedDeltaSource:
    """Reconstruct selected tensors as anchor plus delta and retain other endpoint tensors.

    Verify payload hashes and checkpoint bindings when loading.
    """

    def __init__(self, anchor, endpoint, root, *, receipt_sha256, expected_binding):
        self.anchor, self.endpoint, self.root = anchor, endpoint, Path(root)
        if file_sha256(self.root / "receipt.json") != receipt_sha256:
            raise ValueError("receipt hash mismatch")
        if json.loads((self.root / "DONE.json").read_text())["receipt_sha256"] != receipt_sha256:
            raise ValueError("terminal mismatch")
        self.receipt = json.loads((self.root / "receipt.json").read_text())
        # Normalize tuples to JSON lists before comparing bindings.
        if self.receipt["binding"] != json.loads(json.dumps(expected_binding)):
            raise ValueError("preparation binding mismatch")
        if file_sha256(self.root / "payload.safetensors") != self.receipt["payload_sha256"]:
            raise ValueError("payload hash mismatch")
        if not set(self.receipt["matrices"]) <= (anchor.keys & endpoint.keys):
            raise ValueError("missing aligned projection")

    @property
    def keys(self):
        return self.endpoint.keys

    def tensor(self, key):
        if key not in self.receipt["matrices"]:
            return self.endpoint.tensor(key)
        meta = self.receipt["matrices"][key]
        from .bitdelta import reconstruct_bitdelta
        from .delta_come import reconstruct_delta_come

        with safe_open(self.root / "payload.safetensors", framework="pt", device="cpu") as f:
            get = f.get_tensor
            delta = (
                reconstruct_bitdelta(meta, get)
                if meta["method"] == "bitdelta"
                else reconstruct_delta_come(meta, get)
            )
        base, target = self.anchor.tensor(key), self.endpoint.tensor(key)
        if base.shape != delta.shape or target.shape != delta.shape:
            raise ValueError("projection shape mismatch")
        return (base.float() + delta).to(target.dtype)
