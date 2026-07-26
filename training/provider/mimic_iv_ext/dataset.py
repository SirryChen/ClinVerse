"""
PyTorch Dataset for MIMIC-IV lab prediction training.

Reads JSONL from prepare_data.py, pretokenizes once into an on-disk HuggingFace
Dataset cache (memory-mapped), and serves tensors directly during training.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

log = logging.getLogger(__name__)

_TOKENIZER: PreTrainedTokenizer | None = None
_TOKENIZER_PATH: str | None = None


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _wait_for_cache(cache_dir: Path, timeout: float = 86_400) -> None:
    marker = cache_dir / "dataset_info.json"
    if marker.exists():
        return
    log.info("Rank %d waiting for tokenize cache: %s", _ddp_local_rank(), cache_dir)
    start = time.time()
    while not marker.exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for tokenize cache: {cache_dir}")
        time.sleep(10)


def _cache_dir_for(
    jsonl_path: Path,
    model_path: str,
    max_length: int,
    max_samples: Optional[int],
    cache_root: Path,
) -> Path:
    stat = jsonl_path.stat()
    payload = (
        f"{jsonl_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:"
        f"{model_path}:{max_length}:{max_samples}"
    )
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return cache_root / key


def _init_worker_tokenizer(model_path: str, trust_remote_code: bool) -> PreTrainedTokenizer:
    global _TOKENIZER, _TOKENIZER_PATH
    if _TOKENIZER is None or _TOKENIZER_PATH != model_path:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
        _TOKENIZER_PATH = model_path
    return _TOKENIZER


def _apply_chat_template(tokenizer, messages, *, enable_thinking: bool = False, **kwargs) -> str:
    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_template_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    if "enable_thinking" in sig.parameters or accepts_template_kwargs:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _tokenize_messages(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    max_length: int,
) -> dict[str, list[int]]:
    full_text = _apply_chat_template(
        tokenizer, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = _apply_chat_template(
        tokenizer, prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    # 先对 label 单独分词（不做 truncation）——lab 值通常只有 3-15 个 token，不会超长
    label_text = full_text[len(prompt_text):]
    label_enc  = tokenizer(label_text, add_special_tokens=False)
    label_ids  = label_enc["input_ids"]

    # prompt 截断到 max_length - len(label)，保证 label 永远不丢失
    max_prompt_len = max(1, max_length - len(label_ids))
    prompt_enc  = tokenizer(prompt_text, max_length=max_prompt_len, truncation=True)
    prompt_ids  = prompt_enc["input_ids"]
    prompt_mask = prompt_enc["attention_mask"]

    input_ids      = prompt_ids + label_ids
    attention_mask = prompt_mask + [1] * len(label_ids)
    # 只对 label 部分计算 loss
    labels = [-100] * len(prompt_ids) + label_ids

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "length":         len(input_ids),
    }


def _build_tokenize_cache(
    jsonl_path: Path,
    cache_dir: Path,
    model_path: str,
    max_length: int,
    max_samples: Optional[int],
    num_proc: int,
    trust_remote_code: bool,
) -> None:
    from datasets import load_dataset

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        return

    building = cache_dir.parent / f"{cache_dir.name}.building"
    building.touch()
    try:
        log.info("Building tokenize cache → %s", cache_dir)
        raw = load_dataset("json", data_files=str(jsonl_path), split="train")
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        model_path_str = str(model_path)
        max_len = max_length
        trust = trust_remote_code

        def _map_fn(example: dict) -> dict[str, list[int]]:
            tok = _init_worker_tokenizer(model_path_str, trust)
            return _tokenize_messages(tok, example["messages"], max_len)

        tokenized = raw.map(
            _map_fn,
            remove_columns=raw.column_names,
            num_proc=max(1, num_proc),
            desc="Pretokenizing",
        )
        tokenized.save_to_disk(str(cache_dir))
        log.info("Saved %d tokenized examples → %s", len(tokenized), cache_dir)
    finally:
        building.unlink(missing_ok=True)


class LabPredictionDataset(Dataset):
    """Memory-mapped, pretokenized dataset for causal LM training."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        cache_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        num_proc: int = 4,
        trust_remote_code: bool = True,
    ) -> None:
        from datasets import load_from_disk

        jsonl_path = Path(jsonl_path)
        model_path = str(model_path or getattr(tokenizer, "name_or_path", ""))
        cache_root = Path(cache_dir or "./output/tokenize_cache")
        self.cache_dir = _cache_dir_for(jsonl_path, model_path, max_length, max_samples, cache_root)

        if not (self.cache_dir / "dataset_info.json").exists():
            if _ddp_local_rank() == 0:
                _build_tokenize_cache(
                    jsonl_path=jsonl_path,
                    cache_dir=self.cache_dir,
                    model_path=model_path,
                    max_length=max_length,
                    max_samples=max_samples,
                    num_proc=num_proc,
                    trust_remote_code=trust_remote_code,
                )
            else:
                _wait_for_cache(self.cache_dir)

        log.info("Loading tokenized cache from %s", self.cache_dir)
        self.ds = load_from_disk(str(self.cache_dir))
        self.lengths: list[int] = self.ds["length"]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.ds[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


def make_collator(tokenizer: PreTrainedTokenizer, max_length: int = 2048):
    """Return a collate_fn that left-pads a batch to the same length."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = min(max(b["input_ids"].size(0) for b in batch), max_length)

        input_ids_list, attn_list, labels_list = [], [], []
        for b in batch:
            seq = b["input_ids"][:max_len]
            pad_len = max_len - seq.size(0)
            input_ids_list.append(torch.cat([torch.full((pad_len,), pad_id), seq]))
            attn_list.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), b["attention_mask"][:max_len]]))
            lbl = b["labels"][:max_len]
            labels_list.append(torch.cat([torch.full((pad_len,), -100), lbl]))

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_list),
            "labels": torch.stack(labels_list),
        }

    return collate
