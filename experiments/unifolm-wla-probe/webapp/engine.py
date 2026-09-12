"""Load a UnifoLM ER checkpoint and answer one image+prompt request at a time.

ER-1 and ER-Flow are Qwen3-VL-4B derivatives, so stock transformers loads them.
Only one checkpoint is kept resident: both are ~9 GB in bf16 and the box has 24 GB,
so a model switch frees the previous one instead of risking an OOM mid-request.
"""

import gc
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MAX_NEW_TOKENS = 96


class ModelRunner:
    def __init__(self, model_dirs):
        self.model_dirs = dict(model_dirs)
        self._loaded = None
        self._processor = None
        self._model = None
        self.load_seconds = None

    def _free(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._processor = None
        self._loaded = None
        gc.collect()
        torch.cuda.empty_cache()

    def ensure(self, name):
        if name not in self.model_dirs:
            raise KeyError(f"unknown model {name!r}")
        if self._loaded == name:
            return
        self._free()
        started = time.time()
        processor = AutoProcessor.from_pretrained(self.model_dirs[name])
        kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16}
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_dirs[name], **kwargs)
        except TypeError:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_dirs[name], **kwargs)
        model.eval()
        self._processor = processor
        self._model = model
        self._loaded = name
        self.load_seconds = round(time.time() - started, 1)

    @property
    def resident(self):
        return self._loaded

    def status(self):
        allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        return {
            "resident": self._loaded,
            "load_seconds": self.load_seconds,
            "cuda_allocated_gb": round(allocated, 2),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

    def generate(self, name, image, prompt, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, assistant_prefix=""):
        self.ensure(name)
        processor, model = self._processor, self._model
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if assistant_prefix:
            # The chat template closes an assistant turn with <|im_end|>, so a priming
            # token has to be appended to the rendered string, not sent as a message.
            text += assistant_prefix
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]
        started = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )
        elapsed = time.time() - started
        new_ids = out[0][prompt_len:].tolist()
        tokenizer = processor.tokenizer
        return {
            "text": tokenizer.decode(new_ids, skip_special_tokens=False),
            "text_plain": tokenizer.decode(new_ids, skip_special_tokens=True),
            "n_new_tokens": len(new_ids),
            "hit_max_new_tokens": len(new_ids) >= max_new_tokens,
            "prompt_tokens": prompt_len,
            "seconds": round(elapsed, 2),
            "tokens_per_second": round(len(new_ids) / elapsed, 1) if elapsed > 0 else None,
            "token_ids": new_ids,
            "prompt_sent": text,
        }
