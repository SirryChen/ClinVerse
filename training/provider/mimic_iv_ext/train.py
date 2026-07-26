"""
LoRA SFT training for MIMIC-IV-Ext lab result prediction.

Supports two modes based on config.bnb.load_in_4bit:
  - false (default for 100GB+ VRAM): load model in bf16, no quantization
  - true  (low-VRAM fallback):       QLoRA with 4-bit BitsAndBytes

Usage:
    python train.py [--config config.yaml]

Prerequisites:
    pip install transformers peft accelerate datasets torch pyyaml tqdm
    pip install bitsandbytes   # only needed when load_in_4bit=true

Run prepare_data.py first to build output/train.jsonl and output/test.jsonl.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import yaml

log = logging.getLogger(__name__)


def _check_deps(use_bnb: bool) -> None:
    required = ["transformers", "peft", "accelerate", "datasets"]
    if use_bnb:
        required.append("bitsandbytes")
    missing = [p for p in required if not _importable(p)]
    if missing:
        sys.exit(
            f"Missing packages: {', '.join(missing)}\n"
            "Install with: pip install " + " ".join(missing)
        )


def _importable(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_path = Path(cfg["model"]["name_or_path"])
    if not model_path.exists():
        sys.exit(
            f"Model weights not found: {model_path.resolve()}\n"
            "Download with: huggingface-cli download Qwen/Qwen3.5-... "
            f"--local-dir {model_path}"
        )

    trust = cfg["model"].get("trust_remote_code", True)
    dtype_name = cfg["model"].get("torch_dtype", "bfloat16")
    compute_dtype = getattr(torch, dtype_name)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank != -1
    device_map = None if is_ddp else "auto"
    if is_ddp:
        torch.cuda.set_device(local_rank)
        log.info("DDP mode: local_rank=%d, device=cuda:%d", local_rank, local_rank)

    log.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = cfg.get("bnb", {})
    use_4bit = bnb_cfg.get("load_in_4bit", False)

    if use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=bnb_cfg.get("bnb_4bit_quant_type", "nf4"),
        )
        log.info("Loading model in 4-bit (QLoRA) from %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=trust,
            torch_dtype=compute_dtype,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False),
        )
    else:
        log.info("Loading model in %s (no quantization) from %s", dtype_name, model_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device_map,
            trust_remote_code=trust,
            torch_dtype=compute_dtype,
        )

    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_datasets(cfg: dict, tokenizer):
    from dataset import LabPredictionDataset

    d_cfg = cfg["data"]
    max_length = cfg["model"].get("max_length", 4096)
    train_ds = LabPredictionDataset(
        jsonl_path=d_cfg["train_file"],
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=d_cfg.get("max_samples"),
        cache_dir=d_cfg.get("tokenize_cache_dir", "./output/tokenize_cache"),
        model_path=cfg["model"]["name_or_path"],
        num_proc=d_cfg.get("pretokenize_num_proc", 4),
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    log.info("Train: %d examples (pretokenized cache)", len(train_ds))
    return train_ds


def _sampling_kwargs(t_cfg: dict) -> dict:
    """Map config sampling strategy to TrainingArguments (v4/v5 compatible)."""
    import inspect
    from transformers import TrainingArguments

    strategy = t_cfg.get("train_sampling_strategy", "random")
    params = inspect.signature(TrainingArguments.__init__).parameters

    if "train_sampling_strategy" in params:
        return {"train_sampling_strategy": strategy}
    if strategy == "group_by_length" and "group_by_length" in params:
        return {"group_by_length": True}
    return {}


def _warmup_kwargs(t_cfg: dict, num_training_steps: int) -> dict:
    """Prefer warmup_steps on transformers v5.2+."""
    import inspect
    from transformers import TrainingArguments

    if t_cfg.get("warmup_steps") is not None:
        return {"warmup_steps": t_cfg["warmup_steps"]}

    ratio = t_cfg.get("warmup_ratio")
    if ratio is None:
        return {}

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": max(1, int(num_training_steps * ratio))}


def _trainer_processing_kwargs(tokenizer) -> dict:
    """Support transformers v5 (processing_class) and v4 (tokenizer)."""
    import inspect
    from transformers import Trainer

    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}


def _length_grouped_sampler(train_dataset, lengths, args):
    """Build LengthGroupedSampler compatible with transformers v4 and v5."""
    import inspect
    from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, LengthGroupedSampler

    model_input_name = "input_ids"
    batch_size = args.train_batch_size * args.gradient_accumulation_steps
    sig = inspect.signature(LengthGroupedSampler.__init__)
    first = next(iter(sig.parameters.values()))
    v5_style = first.name == "batch_size"

    if v5_style:
        if args.world_size <= 1:
            return LengthGroupedSampler(
                batch_size,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        return DistributedLengthGroupedSampler(
            batch_size,
            dataset=train_dataset,
            num_replicas=args.world_size,
            rank=args.process_index,
            lengths=lengths,
            model_input_name=model_input_name,
        )

    if args.world_size <= 1:
        return LengthGroupedSampler(
            train_dataset,
            batch_size,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    return DistributedLengthGroupedSampler(
        train_dataset,
        batch_size,
        num_replicas=args.world_size,
        rank=args.process_index,
        lengths=lengths,
        model_input_name=model_input_name,
    )


class LabTrainer:
    """Lazy wrapper: subclass Trainer only when instantiated."""

    @staticmethod
    def create(**kwargs):
        from transformers import Trainer

        class _LabTrainer(Trainer):
            def _get_train_sampler(self, train_dataset=None):
                if train_dataset is None:
                    train_dataset = self.train_dataset

                strategy = getattr(self.args, "train_sampling_strategy", None)
                if strategy is None and getattr(self.args, "group_by_length", False):
                    strategy = "group_by_length"
                if strategy != "group_by_length":
                    return super()._get_train_sampler(train_dataset)

                lengths = getattr(train_dataset, "lengths", None)
                if lengths is None:
                    log.warning(
                        "train_sampling_strategy=group_by_length but dataset has no "
                        "precomputed lengths; falling back to default sampler"
                    )
                    return super()._get_train_sampler(train_dataset)

                return _length_grouped_sampler(train_dataset, lengths, self.args)

        return _LabTrainer(**kwargs)


def run_training(cfg: dict, model, tokenizer, train_ds) -> None:
    from transformers import TrainingArguments
    from dataset import make_collator

    t_cfg = cfg["training"]
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = (
        t_cfg.get("per_device_train_batch_size", 16)
        * world_size
        * t_cfg.get("gradient_accumulation_steps", 1)
    )
    steps_per_epoch = max(1, math.ceil(len(train_ds) / effective_batch))
    num_training_steps = steps_per_epoch * t_cfg.get("num_train_epochs", 3)

    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 16),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", False),
        learning_rate=t_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=t_cfg.get("logging_steps", 20),
        eval_strategy="no",
        save_strategy="steps",
        save_steps=t_cfg.get("save_steps", 500),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        fp16=t_cfg.get("fp16", False),
        bf16=t_cfg.get("bf16", True),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 8),
        remove_unused_columns=False,
        report_to=t_cfg.get("report_to", "none"),
        ddp_find_unused_parameters=False,
        **_warmup_kwargs(t_cfg, num_training_steps),
        **_sampling_kwargs(t_cfg),
    )

    collator = make_collator(tokenizer, max_length=cfg["model"].get("max_length", 4096))

    trainer = LabTrainer.create(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        **_trainer_processing_kwargs(tokenizer),
    )

    log.info(
        "Starting training — %d steps/epoch, %d total steps, sampling=%s",
        steps_per_epoch,
        num_training_steps,
        t_cfg.get("train_sampling_strategy", "random"),
    )
    trainer.train()

    adapter_dir = Path(t_cfg["output_dir"]) / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("Saved final adapter → %s", adapter_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_4bit = cfg.get("bnb", {}).get("load_in_4bit", False)
    _check_deps(use_4bit)

    for key in ("output_dir",):
        Path(cfg["data"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["data"].get("tokenize_cache_dir", "./output/tokenize_cache")).mkdir(parents=True, exist_ok=True)

    train_path = Path(cfg["data"]["train_file"])
    if not train_path.exists():
        sys.exit(f"Training data not found: {train_path}\nRun: python prepare_data.py")

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds = build_datasets(cfg, tokenizer)
    run_training(cfg, model, tokenizer, train_ds)


if __name__ == "__main__":
    main()
