"""Collect endpoint completions and clean-sequence projection inputs for output LS.

Completion records and per-domain activation reservoirs are saved separately.
"""

from collections import Counter
import hashlib
import math
from pathlib import Path
import os

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import (
    atomic_json,
    content_sha,
    file_sha,
    finish,
    read_json,
    run_directory,
    runtime,
    verify_done,
)
from .checkpoints import Checkpoint, identity
from .download import catalog, dataset_cache, snapshot
from .merge.output_ls import exact_output_terms, positive_output_scale, terms_error
from .merge.support import signed_top_support

DOMAINS = ("code", "math", "stem", "chat")
GRID = tuple(i / 100 for i in range(5, 100, 5))


def validate_panel(rows):
    if Counter((r["domain"], r["split"]) for r in rows) != Counter(
        {(d, s): n for d in DOMAINS for s, n in [("fit", 32), ("validation", 8)]}
    ):
        raise ValueError("Requires code/math/stem/chat × 32 fit + 8 validation")
    keys = [content_sha(r["messages"]) for r in rows]
    if len(set(keys)) != 160 or any(any(m["role"] != "user" for m in r["messages"]) for r in rows):
        raise ValueError("Calibration must be unique disjoint user-only messages")


def build_panel(output, seed=20260908):
    """Build a seeded, disjoint calibration and validation panel."""
    from datasets import load_dataset

    spec = catalog()["calibration"]
    rows, seen = [], set()
    for d, domain in enumerate(DOMAINS):
        stream = load_dataset(
            spec["repo_id"],
            data_files=f"data/{domain}-*.parquet",
            revision=spec["revision"],
            split="train",
            streaming=True,
            cache_dir=dataset_cache(),
        ).shuffle(seed=seed + d, buffer_size=10000)
        selected = []
        for item in stream:
            users = [
                m
                for m in item["messages"]
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ]
            if not users:
                continue
            messages = [{"role": "user", "content": "\n\n".join(m["content"] for m in users)}]
            key = content_sha(messages)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                dict(
                    id=item.get("uuid", key),
                    domain=domain,
                    split="fit" if len(selected) < 32 else "validation",
                    messages=messages,
                )
            )
            if len(selected) == 40:
                break
        rows.extend(selected)
    validate_panel(rows)
    with run_directory(
        output,
        dict(
            dataset=spec, seed=seed, sampling="seeded_stream_shuffle_buffer10000", runtime=runtime()
        ),
    ) as root:
        atomic_json(root / "panel.json", rows)
        finish(root, ["manifest.json", "panel.json"])
    return rows


def projection_keys(model):
    return sorted(
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(getattr(module, "weight", None), torch.nn.Parameter)
        and module.weight.ndim == 2
        and not isinstance(module, torch.nn.Embedding)
        and not any(x in name for x in ("lm_head", "embed", "output_layer"))
    )


def representative(key, family):
    if family == "llada" and (
        ".mlp.experts." in key or ".mlp.shared_experts." in key or key.endswith(".mlp.gate.weight")
    ):
        return key.split(".mlp.")[0] + ".mlp"
    return key[:-7]


def projection_rows(bank, key, rep, endpoint, device):
    """Compute projection inputs, including auxiliary MoE expert inputs for statistics."""
    x = bank.to(device)
    if rep.endswith(".mlp") and key.endswith(".down_proj.weight"):
        prefix = key.removesuffix("down_proj.weight")
        gate = endpoint.tensor(prefix + "gate_proj.weight").to(device)
        up = endpoint.tensor(prefix + "up_proj.weight").to(device)
        x = x.to(gate.dtype)
        x = torch.nn.functional.silu(
            torch.nn.functional.linear(x, gate)
        ) * torch.nn.functional.linear(x, up)
    return x.float()


def static_forward(model, family, ids):
    p = torch.arange(ids.shape[1], device=ids.device)
    common = dict(input_ids=ids, position_ids=p[None], past_key_values=None, use_cache=False)
    if family == "fast":
        return model.model(
            **common,
            cache_position=p,
            block_size=32,
            block_past_key_values=None,
            use_block_cache=False,
            update_past_key_values=False,
        )
    if family in ("sdar", "llada"):
        block = catalog()["models"][family]["block_size"]
        allowed = p[:, None] // block >= p[None, :] // block
        mask = (
            allowed
            if family == "sdar"
            else torch.zeros_like(allowed, dtype=model.dtype).masked_fill(~allowed, -torch.inf)
        )
        extra = (
            dict(store_kv=False, cache_position=p) if family == "sdar" else dict(return_dict=True)
        )
        return model.model(**common, attention_mask=mask[None, None], **extra)
    return model.model(**common, cache_position=p, return_dict=True)


class Reservoir:
    """Uniform row reservoir using independent random priorities."""

    def __init__(self, cap, seed):
        self.cap, self.rng = cap, np.random.default_rng(seed)
        self.rows, self.priority, self.seen = None, np.empty(0), 0

    def add(self, x):
        x = x.detach().reshape(-1, x.shape[-1])
        scores = self.rng.random(len(x))
        self.seen += len(x)
        # Transfer selected reservoir candidates to CPU.
        chosen = np.argsort(scores)[: self.cap]
        values = x[torch.as_tensor(chosen, device=x.device)].cpu()
        priorities = np.concatenate([self.priority, scores[chosen]])
        rows = values if self.rows is None else torch.cat([self.rows, values])
        keep = np.argsort(priorities)[: self.cap]
        self.rows, self.priority = rows[torch.as_tensor(keep)], priorities[keep]


def collect(model, panel, output, *, device="cuda:0", budget=1024, seed=20260908):
    from transformers import AutoModelForCausalLM
    from .models import Generator

    rows = read_json(panel)
    validate_panel(rows)
    contract = dict(
        model=model,
        checkpoints=catalog()["models"][model],
        endpoint_files=identity(snapshot(model, "diffusion")),
        panel_sha256=file_sha(panel),
        population="diffusion_native_static_clean",
        fit_rows_per_domain=384,
        validation_rows_per_domain=192,
        seed=seed,
        budget=budget,
        block_origin=0,
        eos_included=True,
        extra_padding=False,
        runtime=runtime(),
    )
    with run_directory(output, contract) as root:
        if (root / "DONE.json").exists():
            return verify_done(root)
        generator = Generator(model, device=device)
        for i, row in enumerate(rows):
            path = root / "completions" / f"{i:05d}.json"
            if path.exists():
                if read_json(path)["item_sha256"] != content_sha(row):
                    raise ValueError("Saved completion identity drift")
                continue
            ids = generator.encode(row["messages"])
            torch.manual_seed(seed + i)
            generated = generator.generate(ids, budget)
            atomic_json(
                path, dict(row, prompt_ids=ids, completion=generated, item_sha256=content_sha(row))
            )
        # Load the HF backbone for projection-input hooks.
        if generator.model is None:
            generator.close()
            del generator
            torch.cuda.empty_cache()
            backbone = AutoModelForCausalLM.from_pretrained(
                snapshot(model, "diffusion"),
                torch_dtype=getattr(torch, catalog()["models"][model]["dtype"]),
                trust_remote_code=True,
                device_map=device,
            ).eval()
        else:
            backbone = generator.model
        inventory = projection_keys(backbone)
        modules = dict(backbone.named_modules())
        representatives = {k: representative(k, model) for k in inventory}
        spec = catalog()["models"][model]
        files = ["manifest.json"] + [f"completions/{i:05d}.json" for i in range(len(rows))]
        # Save an activation reservoir for each domain and split.
        for split in ("fit", "validation"):
            for domain in DOMAINS:
                path = root / "activations" / f"{split}-{domain}.safetensors"
                meta = path.with_suffix(".json")
                if meta.exists():
                    if file_sha(path) != read_json(meta)["sha256"]:
                        raise ValueError("Reservoir artifact drift")
                    files += [str(path.relative_to(root)), str(meta.relative_to(root))]
                    continue
                reservoirs = {
                    rep: Reservoir(
                        384 if split == "fit" else 192,
                        int(
                            hashlib.sha256(f"{seed}:{split}:{domain}:{rep}".encode()).hexdigest()[
                                :16
                            ],
                            16,
                        ),
                    )
                    for rep in sorted(set(representatives.values()))
                }
                handles = []
                for key in reservoirs:
                    handles.append(
                        modules[key].register_forward_pre_hook(
                            lambda module, args, k=key: reservoirs[k].add(args[0])
                        )
                    )
                try:
                    for i, row in enumerate(rows):
                        if (row["domain"], row["split"]) != (domain, split):
                            continue
                        saved = read_json(root / "completions" / f"{i:05d}.json")
                        tokens = saved["completion"]["token_ids"]
                        end = next(
                            (j + 1 for j, t in enumerate(tokens) if t in spec["stop_ids"]),
                            len(tokens),
                        )
                        ids = saved["prompt_ids"] + tokens[:end]
                        if spec["mask_id"] in ids:
                            raise ValueError("Mask remains in static clean sequence")
                        with torch.inference_mode():
                            static_forward(backbone, model, torch.tensor([ids], device=device))
                finally:
                    for h in handles:
                        h.remove()
                missing = [
                    k for k, r in reservoirs.items() if r.rows is None or len(r.rows) < r.cap
                ]
                if missing:
                    atomic_json(
                        root / "coverage.json",
                        dict(
                            domain=domain,
                            split=split,
                            missing=missing,
                            policy="fail_closed_no_implicit_expert_fallback",
                        ),
                    )
                    raise ValueError(
                        f"Insufficient projection rows ({len(missing)}); see coverage.json"
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {k: r.rows.contiguous() for k, r in reservoirs.items()}, str(path) + ".tmp"
                )
                os.replace(str(path) + ".tmp", path)
                atomic_json(
                    meta,
                    dict(
                        sha256=file_sha(path),
                        counts={
                            k: dict(seen=r.seen, selected=len(r.rows))
                            for k, r in reservoirs.items()
                        },
                    ),
                )
                files += [str(path.relative_to(root)), str(meta.relative_to(root))]
        atomic_json(
            root / "inventory.json",
            dict(
                keys=inventory,
                representatives=representatives,
                expert_population="offline_all_expert_bank" if model == "llada" else None,
            ),
        )
        finish(root, files + ["inventory.json"])
        return dict(projections=len(inventory), items=160)


def fit(model, collection, output, *, alphas=GRID, device="cpu"):
    alphas = tuple(sorted(set(alphas)))
    if not alphas or any(not math.isfinite(a) or not 0 <= a < 1 for a in alphas):
        raise ValueError("Alpha candidates must be finite values in [0,1)")
    # Disable TF32 for projection-output least-squares accumulation.
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(collection)
    verify_done(root)
    original = read_json(root / "manifest.json")
    if original["model"] != model or original["checkpoints"] != catalog()["models"][model]:
        raise ValueError("Collection endpoint/anchor identity mismatch")
    endpoint_files = identity(snapshot(model, "diffusion"))
    if original.get("endpoint_files") != endpoint_files:
        raise ValueError("Collection endpoint file identity mismatch")
    contract = dict(
        model=model,
        checkpoints=catalog()["models"][model],
        base_files=identity(snapshot(model, "ar")),
        endpoint_files=endpoint_files,
        collection_sha256=content_sha(read_json(root / "DONE.json")),
        alphas=list(alphas),
        objective="full_projection_output_ls_equal_domains_nonnegative",
        runtime=runtime(),
    )
    inventory = read_json(root / "inventory.json")
    keys = inventory["keys"]
    with run_directory(output, contract) as out:
        if (out / "DONE.json").exists():
            return verify_done(out)
        scores = {alpha: [0.0, 0.0, 0.0, 0.0] for alpha in alphas}
        scale_paths = []
        with (
            Checkpoint(snapshot(model, "ar")) as base,
            Checkpoint(snapshot(model, "diffusion")) as end,
        ):
            for i, key in enumerate(keys):
                path = out / "per_projection" / f"{i:05d}.safetensors"
                report = path.with_suffix(".json")
                if report.exists():
                    done = read_json(report)
                    if done["key"] != key or file_sha(path) != done["sha256"]:
                        raise ValueError("Projection fit resume drift")
                else:
                    delta = (end.tensor(key).float() - base.tensor(key).float()).to(device)
                    data = {}
                    for split in ("fit", "validation"):
                        data[split] = []
                        for domain in DOMAINS:
                            with safe_open(
                                root / "activations" / f"{split}-{domain}.safetensors",
                                framework="pt",
                                device="cpu",
                            ) as f:
                                rep = inventory["representatives"][key]
                                data[split].append(
                                    projection_rows(f.get_tensor(rep), key, rep, end, device)
                                )
                    saved, records = {}, {}
                    for alpha in alphas:
                        code = signed_top_support(delta, alpha) * delta.sign()
                        terms = {
                            s: sum(exact_output_terms(x, delta, code) for x in xs) / 4
                            for s, xs in data.items()
                        }
                        scale = positive_output_scale(terms["fit"])
                        saved[str(alpha)] = scale.float().contiguous()
                        records[str(alpha)] = [
                            float(terms_error(terms["fit"], scale)),
                            float(terms["fit"][2].sum()),
                            float(terms_error(terms["validation"], scale)),
                            float(terms["validation"][2].sum()),
                        ]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(saved, str(path) + ".tmp")
                    os.replace(str(path) + ".tmp", path)
                    done = dict(key=key, scores=records, sha256=file_sha(path))
                    atomic_json(report, done)
                scale_paths.append(path)
                for alpha in alphas:
                    scores[alpha] = [
                        a + b for a, b in zip(scores[alpha], done["scores"][str(alpha)])
                    ]
        if any(v[1] <= 0 for v in scores.values()):
            raise ValueError("Undefined zero task-vector energy")
        alpha = min(alphas, key=lambda a: scores[a][0] / scores[a][1])
        tensors = {}
        for key, path in zip(keys, scale_paths):
            with safe_open(path, framework="pt", device="cpu") as f:
                tensors[key] = f.get_tensor(str(alpha))
        save_file(tensors, str(out / "scales.safetensors"))
        result = dict(
            alpha=alpha,
            selection="fit_only_smallest_alpha_tie",
            grid={
                str(a): dict(fit_error=v[0] / v[1], validation_error=v[2] / v[3] if v[3] else None)
                for a, v in scores.items()
            },
        )
        atomic_json(out / "selection.json", result)
        finish(out, ["manifest.json", "selection.json", "scales.safetensors"])
        return result
