"""QLoRA supervised fine-tune of Qwen2.5-7B-Instruct into the HSK-5 tutor.

Runs on a CUDA GPU (Colab) — bitsandbytes 4-bit is CUDA-only. It loads the
chat-format data from data/*.jsonl, attaches a LoRA adapter to the 4-bit base,
trains with TRL's SFTTrainer, and saves the adapter to outputs/.

Usage (on Colab):
    python train.py                 # full run, hyperparams from config.TRAIN
    python train.py --max-steps 20  # quick sanity run (a minute or two)

The dataset is "conversational" (each row has a `messages` list), so SFTTrainer
applies the tokenizer's chat template automatically — we don't format prompts by
hand. We only ever compute loss on the assistant turns.
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config as c


def build_bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization for the frozen base (QLoRA)."""
    t = c.TRAIN
    return BitsAndBytesConfig(
        load_in_4bit=t.load_in_4bit,
        bnb_4bit_quant_type=t.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=t.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, t.bnb_4bit_compute_dtype),
    )


def build_lora_config() -> LoraConfig:
    t = c.TRAIN
    return LoraConfig(
        r=t.lora_r,
        lora_alpha=t.lora_alpha,
        lora_dropout=t.lora_dropout,
        target_modules=list(t.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT for the HSK-5 tutor.")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps for a quick sanity run")
    ap.add_argument("--output-dir", type=str, default=str(c.OUTPUT_DIR), help="where to save the adapter")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU found. train.py needs one (bitsandbytes 4-bit is CUDA-only) — "
            "run it on Colab, not the Mac. See train_colab.ipynb."
        )

    t = c.TRAIN
    torch.manual_seed(t.seed)

    # --- data: conversational jsonl (each row has `messages`) -------------- #
    ds = load_dataset(
        "json",
        data_files={"train": str(c.TRAIN_FILE), "eval": str(c.EVAL_FILE)},
    )
    print(f"loaded {len(ds['train'])} train / {len(ds['eval'])} eval examples")

    # --- tokenizer -------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(c.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen has no pad token by default

    # --- 4-bit base ------------------------------------------------------- #
    model = AutoModelForCausalLM.from_pretrained(
        c.BASE_MODEL,
        quantization_config=build_bnb_config(),
        device_map="auto",
        torch_dtype=getattr(torch, t.bnb_4bit_compute_dtype),
    )
    model.config.use_cache = False  # required with gradient checkpointing

    # --- SFT config ------------------------------------------------------- #
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=t.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum_steps,
        learning_rate=t.lr,
        lr_scheduler_type=t.lr_scheduler,
        warmup_ratio=t.warmup_ratio,
        logging_steps=t.logging_steps,
        max_seq_length=t.max_seq_len,
        bf16=True,
        optim="paged_adamw_8bit",           # memory-friendly optimizer for QLoRA
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",                   # no wandb prompt
        seed=t.seed,
    )

    # SFTTrainer applies the chat template, masks non-assistant tokens, and
    # (given peft_config + a quantized model) wraps the base with LoRA and
    # preps it for k-bit training.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        peft_config=build_lora_config(),
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(args.output_dir)      # adapter weights
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nadapter saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
