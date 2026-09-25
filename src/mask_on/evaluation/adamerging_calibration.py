"""Compute differentiable first-denoising-state entropy for each model family."""

import hashlib

import torch
from torch.nn.utils import parametrize


def panel_audit(rows):
    from collections import Counter

    counts = Counter((r["split"], r["domain"]) for r in rows)
    expected = {
        (s, d): n
        for s, n in [("fit", 32), ("validation", 8)]
        for d in ("code", "math", "stem", "chat")
    }
    if dict(counts) != expected or len({r["prompt_sha256"] for r in rows}) != 160:
        raise ValueError("panel must contain disjoint 128-fit / 32-validation user prompts")
    if any(not r["messages"] for r in rows) or any(
        m["role"] != "user" or not isinstance(m["content"], str) or not m["content"].strip()
        for r in rows
        for m in r["messages"]
    ):
        raise ValueError("non-user calibration content")
    import json

    if len({json.dumps(r["messages"], sort_keys=True) for r in rows}) != 160:
        raise ValueError("duplicate prompts across calibration/selection items")
    return counts


def block_mask(length, block_size, *, device, dtype):
    positions = torch.arange(length, device=device)
    allowed = positions[:, None] // block_size >= positions[None, :] // block_size
    return torch.zeros((length, length), device=device, dtype=dtype).masked_fill(
        ~allowed, -torch.inf
    )[None, None]


class NativeFirstState:
    def __init__(self, model, tokenizer, family, *, processor=None, seed=20260909):
        self.model, self.tokenizer, self.family, self.processor, self.seed = (
            model,
            tokenizer,
            family,
            processor,
            seed,
        )
        self.device = next(model.parameters()).device
        self.forward_count = 0

    def prepare(self, rows):
        prepared = []
        for row in rows:
            messages = list(row["messages"])
            if self.family == "fast":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ] + messages
            if self.processor is not None:
                mapping = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                ids = mapping["input_ids"]
            else:
                mapping = None
                if self.family in ("fast", "wedlm8", "sdar8"):
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
                else:
                    ids = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    )
                    if not isinstance(ids, torch.Tensor):
                        ids = ids["input_ids"]
            if ids.shape[0] != 1 or not ids.shape[1]:
                raise ValueError("invalid native prompt")
            prepared.append(
                dict(
                    row=row,
                    ids=ids.cpu(),
                    mapping=mapping,
                    token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                )
            )
        return prepared

    def logits(self, example):
        ids = example["ids"].to(self.device)
        family = self.family
        if family == "sdar8":
            return self._sdar_logits(ids)
        if family == "gemma":
            # Seed the random canvas from the fitting seed and prompt hash.
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), parametrize.cached():
                torch.manual_seed(self.seed ^ int(example["row"]["prompt_sha256"][:8], 16))
                kwargs = {k: v.to(self.device) for k, v in example["mapping"].items()}
                result = self.model(**kwargs, return_dict=True).logits
            self.forward_count += 1
            return result[0]
        mask_id = {
            "dream": 151666,
            "fast": 151665,
            "wedlm8": 151665,
            "sdar8": 151669,
            "llada2": 156895,
        }[family]
        size = {"dream": 512, "fast": 32, "wedlm8": 16, "sdar8": 4, "llada2": 32}[family]
        prefix = ids.shape[1]
        if family == "fast" and prefix > size and prefix % size == 0:
            # Generate the boundary opener with a separate no-grad parameter cache.
            with torch.no_grad(), parametrize.cached():
                opener = (
                    self.model(input_ids=ids, use_cache=False, block_size=size)
                    .logits[:, -1:]
                    .argmax(-1)
                )
            self.forward_count += 1
            ids = torch.cat((ids, opener), 1)
            prefix += 1
        masks = size if family in ("dream", "wedlm8") else size - prefix % size
        x = torch.cat(
            (ids, torch.full((1, masks), mask_id, device=self.device, dtype=ids.dtype)), 1
        )
        positions = torch.arange(x.shape[1], device=self.device)[None]
        kw = dict(input_ids=x, use_cache=False, return_dict=True, position_ids=positions)
        if family == "dream":
            # Use full attention, no padding mask, and the checkpoint's cache default.
            kw = dict(input_ids=x, attention_mask="full", position_ids=None)
        elif family == "fast":
            kw.update(block_size=size, update_past_key_values=False)
        elif family in ("llada2", "sdar8"):
            kw["attention_mask"] = block_mask(
                x.shape[1], size, device=self.device, dtype=torch.bfloat16
            )
            if family == "sdar8":
                # SDAR attention uses True for allowed token pairs.
                kw["attention_mask"] = kw["attention_mask"].eq(0)
        else:
            # Use sequential positions for the initial all-mask WeDLM state.
            kw["attention_mask"] = torch.ones_like(x)
        with parametrize.cached():
            result = self.model(**kw).logits
        self.forward_count += 1
        if family in ("dream", "fast"):
            result = torch.cat((result[:, :1], result[:, :-1]), 1)
        return result[0, prefix : prefix + masks]

    def _sdar_logits(self, ids):
        """Compute logits for the first SDAR block using prefix prefill and a cached tail.

        Each call rebuilds the prefix cache with the current merge coefficients.
        """
        from transformers import DynamicCache

        size, mask_id = 4, 151669
        prefix = ids.shape[1]
        blocks = (prefix + 128 + size - 1) // size
        total = blocks * size
        block_allowed = torch.tril(torch.ones(blocks, blocks, device=self.device))
        allowed = block_allowed.repeat_interleave(size, 0).repeat_interleave(size, 1).unsqueeze(0)
        positions = torch.arange(total, device=self.device).unsqueeze(0)
        x = torch.full((1, total), mask_id, dtype=torch.long, device=self.device)
        x[:, :prefix] = ids
        prefill = prefix // size * size
        cache = DynamicCache()
        with parametrize.cached():
            if prefill:
                self.model(x[:, :prefill], attention_mask=allowed[:, :prefill, :prefill],
                           position_ids=positions[:, :prefill], past_key_values=cache,
                           use_cache=True, store_kv=True)
                self.forward_count += 1
            cur = x[:, prefill:prefill+size].clone()
            result = self.model(cur, attention_mask=allowed[:, prefill:prefill+size, :prefill+size],
                                position_ids=positions[:, prefill:prefill+size], past_key_values=cache,
                                use_cache=True, store_kv=False).logits
            self.forward_count += 1
        return result[0][cur[0] == mask_id]

    def loss(self, example):
        logits = self.logits(example).float()
        if not len(logits) or not torch.isfinite(logits).all():
            raise ValueError("empty/nonfinite calibration logits")
        log_probs = logits.log_softmax(-1)
        return -(log_probs.exp() * log_probs).sum(-1).mean()
