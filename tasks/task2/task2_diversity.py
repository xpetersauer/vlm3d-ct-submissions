"""Safe Hugging Face generation-policy cycling for Task 2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import threading


POLICIES = (
    {"name": "greedy", "do_sample": False, "num_beams": 1},
    {"name": "beam4", "do_sample": False, "num_beams": 4},
    {"name": "nucleus_t070", "do_sample": True, "temperature": 0.70, "top_p": 0.85, "top_k": 40, "num_beams": 1},
    {"name": "nucleus_t095", "do_sample": True, "temperature": 0.95, "top_p": 0.92, "top_k": 60, "num_beams": 1},
    {"name": "nucleus_t115", "do_sample": True, "temperature": 1.15, "top_p": 0.96, "top_k": 100, "num_beams": 1},
)

_state = threading.local()


def _jsonl(path: str | None, row: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def install_generation_patch(*, capture_path: str | None = None) -> None:
    """Cycle policies across consecutive ``generate`` calls.

    The shipped Task-2 code already asks for multiple candidates, but its
    ``do_sample=False`` made temperature/top-p/top-k inert.  This wrapper fixes
    that at the common Transformers API boundary and leaves tensor shapes and
    model weights untouched.
    """
    from transformers.generation.utils import GenerationMixin
    import torch

    if getattr(GenerationMixin.generate, "_vlm3d_diverse", False):
        return
    original = GenerationMixin.generate
    base_seed = int(os.getenv("TASK2_DIVERSITY_SEED", "29001"))

    def patched(self, *args, **kwargs):
        index = int(getattr(_state, "generate_index", 0))
        _state.generate_index = index + 1
        policy = dict(POLICIES[index % len(POLICIES)])
        name = policy.pop("name")
        kwargs.update(policy)
        kwargs.setdefault("repetition_penalty", 1.08)
        kwargs.setdefault("no_repeat_ngram_size", 3)
        if kwargs.get("do_sample"):
            kwargs.pop("num_beam_groups", None)
            kwargs.pop("diversity_penalty", None)
        else:
            for key in ("temperature", "top_p", "top_k", "typical_p"):
                kwargs.pop(key, None)
            return_sequences = int(kwargs.get("num_return_sequences", 1))
            kwargs["num_beams"] = max(int(kwargs.get("num_beams", 1)), return_sequences)
        seed = base_seed + index
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result = original(self, *args, **kwargs)
        try:
            sequences = result.sequences if hasattr(result, "sequences") else result
            digest = hashlib.sha256(sequences.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        except Exception:
            digest = "unavailable"
        record = {"event": "generate", "call": index, "policy": name, "seed": seed, "token_hash": digest}
        _jsonl(capture_path, record)
        if index < 20:
            print(f"[task2-diversity] call={index} policy={name} seed={seed} hash={digest}", flush=True)
        return result

    patched._vlm3d_diverse = True
    patched._vlm3d_original = original
    GenerationMixin.generate = patched


def install_decode_capture(capture_path: str, decoded_sink: list[str]) -> None:
    """Capture decoded candidates without changing tokenizer results."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if getattr(PreTrainedTokenizerBase.decode, "_vlm3d_capture", False):
        return
    original = PreTrainedTokenizerBase.decode

    def patched(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        text = str(value).strip()
        if text:
            decoded_sink.append(text)
            _jsonl(capture_path, {"event": "decoded", "text": text})
        return value

    patched._vlm3d_capture = True
    PreTrainedTokenizerBase.decode = patched
