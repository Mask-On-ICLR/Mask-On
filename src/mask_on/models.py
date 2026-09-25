"""Load native model runtimes and expose prompt encoding and generation."""

from pathlib import Path
import time

from .download import catalog, native_source, snapshot


def render_messages(model, messages):
    system = catalog()["models"][model]["system"]
    return ([{"role": "system", "content": system}] if system else []) + list(messages)


class Generator:
    def __init__(
        self,
        model,
        mode="diffusion",
        checkpoint=None,
        *,
        backend="native",
        threshold=1.0,
        device="cuda:0",
        gpu_memory_utilization=0.8,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.mode, self.backend = model, mode, backend
        self.spec = catalog()["models"][model]
        self.device, self.threshold = device, threshold
        self.path = Path(checkpoint) if checkpoint else snapshot(model, mode)
        token_path = snapshot(model, mode)
        self.tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
        self.model = None
        self._closed = False
        self.max_context = None
        if backend == "vllm":
            if mode != "ar":
                raise ValueError("vLLM is for the native causal AR route only")
            from vllm import LLM

            self.engine = LLM(
                model=str(self.path),
                tokenizer=str(token_path),
                trust_remote_code=True,
                dtype="bfloat16",
                tensor_parallel_size=1,
                max_model_len=8192,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self.max_context = 8192
        elif model == "wedlm" and mode == "diffusion":
            import sys

            sys.path.insert(0, str(native_source("wedlm")))
            from wedlm import LLM

            self.engine = LLM(
                model=str(self.path),
                enforce_eager=True,
                max_num_seqs=1,
                max_num_batched_tokens=4096,
                max_model_len=4096,
                gpu_memory_utilization=gpu_memory_utilization,
                wedlm_window_size=16,
            )
            self.max_context = 4096
        else:
            dtype = getattr(torch, self.spec["dtype"] if mode == "diffusion" else "bfloat16")
            loader = AutoModelForCausalLM
            self.model = loader.from_pretrained(
                self.path, torch_dtype=dtype, device_map=device, trust_remote_code=True
            ).eval()
            self.max_context = getattr(self.model.config, "max_position_embeddings", None)
        self.nfe = 0
        if self.model is not None:
            self.model.register_forward_hook(self._count)

    def _count(self, *_):
        self.nfe += 1

    def encode(self, messages):
        kwargs = (
            {"enable_thinking": False}
            if self.mode == "ar" and self.name in ("wedlm", "sdar")
            else {}
        )
        return self.tokenizer.apply_chat_template(
            render_messages(self.name, messages),
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )

    def generate(self, prompt_ids, budget=1024):
        import torch

        if self._closed or budget < 1 or not prompt_ids:
            raise ValueError("Closed generator or empty prompt/budget")
        if self.max_context is not None and len(prompt_ids) + budget > self.max_context:
            raise ValueError(
                "Prompt plus generation budget exceeds native context; no silent truncation"
            )
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self.nfe = 0
        started = time.perf_counter()
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            if self.backend == "vllm":
                from vllm import SamplingParams

                result = self.engine.generate(
                    [{"prompt_token_ids": prompt_ids}],
                    SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=budget,
                        stop_token_ids=self.spec["stop_ids"],
                    ),
                    use_tqdm=False,
                )
                tokens = list(result[0].outputs[0].token_ids)
            elif self.mode == "ar":
                out = self.model.generate(
                    input_ids=ids,
                    max_new_tokens=budget,
                    do_sample=False,
                    eos_token_id=self.spec["stop_ids"],
                    pad_token_id=self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.spec["stop_ids"][0],
                )
                tokens = out[0, len(prompt_ids) :].tolist()
            elif self.name == "fast":
                out = self.model.generate(
                    ids,
                    tokenizer=self.tokenizer,
                    max_new_tokens=budget,
                    block_size=32,
                    small_block_size=8,
                    threshold=self.threshold,
                    temperature=0.0,
                    top_p=1.0,
                    use_block_cache=False,
                )
                tokens = out[0, len(prompt_ids) :].tolist()
            elif self.name == "llada":
                out = self.model.generate(
                    inputs=ids,
                    eos_early_stop=True,
                    gen_length=budget,
                    block_length=32,
                    steps=32,
                    temperature=0.0,
                )
                # Handle both prefix-inclusive and completion-only native outputs.
                if out.shape[1] >= len(prompt_ids) and torch.equal(out[:, : len(prompt_ids)], ids):
                    out = out[:, len(prompt_ids) :]
                tokens = out[0].tolist()
            elif self.name == "sdar":
                from .evaluation.sdar import sdar_official

                fn = sdar_official(native_source("sdar"))["block_diffusion_generate"]
                out = fn(
                    self.model,
                    self.tokenizer,
                    {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                    gen_length=budget,
                    stopping_criteria_idx=self.spec["stop_ids"],
                    block_length=4,
                    denoising_steps=4,
                    temperature=1.0,
                    top_k=1,
                    top_p=1.0,
                    threshold=1.0,
                    remasking="low_confidence",
                    mask_id=151669,
                )
                if not torch.equal(out[:, : len(prompt_ids)], ids):
                    raise ValueError("SDAR native return changed prefix")
                tokens = out[0, len(prompt_ids) :].tolist()
            else:
                from wedlm import SamplingParams
                from wedlm.engine.scheduler import Scheduler

                if not self.engine.is_finished():
                    raise RuntimeError("Previous WeDLM request is still live")
                self.engine.scheduler = Scheduler(self.engine.model_runner.config)
                result = self.engine.generate(
                    [prompt_ids],
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=budget,
                        top_p=1.0,
                        top_k=0,
                        stop_token_ids=self.spec["stop_ids"],
                        wedlm_entropy_threshold=0.4,
                        wedlm_pos_penalty_factor=0.02,
                    ),
                    use_tqdm=False,
                )
                tokens = list(result[0]["token_ids"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        from .evaluation.common import trim_at_eos

        scored = trim_at_eos(tokens, self.spec["stop_ids"])
        text = self.tokenizer.decode(scored, skip_special_tokens=True)
        return {
            "text": text,
            "token_ids": tokens,
            "seconds": seconds,
            "tokens": len(scored),
            "raw_tokens": len(tokens),
            "nfe": self.nfe if self.model is not None else None,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }

    def close(self):
        if self._closed:
            return
        if self.name == "wedlm" and self.mode == "diffusion" and self.model is None:
            import atexit

            atexit.unregister(self.engine.exit)
            self.engine.exit()
            del self.engine
        self._closed = True
