"""Merge the trained LoRA adapter into the base model → a standalone fp16 model.

QLoRA trains a small adapter on top of a frozen 4-bit base. To serve it we fold
those weights back into a full-precision copy of the base, giving a normal
Transformers model we can then quantize to GGUF for the Mac demo.

Runs wherever there's enough RAM/VRAM to hold the base in fp16 (14B ≈ 28GB; a 7B
was ~15GB) — do it on Colab right after train.py, before the GGUF step.

    python merge.py            # outputs/  ->  outputs/merged/
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as c


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into the base model.")
    ap.add_argument("--adapter-dir", default=str(c.OUTPUT_DIR), help="dir with the trained adapter")
    ap.add_argument("--out", default=str(c.MERGED_DIR), help="where to write the merged fp16 model")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    where = "GPU + disk offload" if use_cuda else "CPU RAM"
    print(f"loading base {c.BASE_MODEL} in fp16 ({where})...")
    base = AutoModelForCausalLM.from_pretrained(
        c.BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # Big-RAM machine (e.g. a 24GB Mac): load straight to CPU — simplest + fastest.
        # Memory-tight CUDA GPU (T4): spread across GPU + CPU + disk.
        device_map="auto" if use_cuda else None,
        offload_folder="offload" if use_cuda else None,
    )
    print(f"applying adapter from {args.adapter_dir} and merging...")
    peft_kwargs = {"offload_folder": "offload"} if use_cuda else {}
    model = PeftModel.from_pretrained(base, args.adapter_dir, **peft_kwargs)
    model = model.merge_and_unload()          # fold LoRA weights into the base

    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(c.BASE_MODEL).save_pretrained(args.out)
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
