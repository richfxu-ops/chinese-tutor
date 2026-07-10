"""Before/after eval: base Qwen2.5-7B vs. the QLoRA-tuned tutor.

Runs on the same CUDA GPU you trained on (Colab). It loads the base once, attaches
the trained adapter, and generates on the held-out eval prompts with the adapter
OFF (base) and ON (tuned) — so the only difference is the fine-tune. Writes a
side-by-side markdown report you can eyeball.

Optionally (--judge, needs ANTHROPIC_API_KEY) it asks Claude to score both on a
small rubric and tallies which one wins — a light, qualitative signal, not a
benchmark.

Usage (on Colab, after train.py):
    python eval.py --n 20
    python eval.py --n 20 --judge
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config as c

RUBRIC_AXES = [
    "level_fit",      # answer stays at/below HSK-5, scaffolds rather than escalates
    "correctness",   # the Chinese is accurate (grammar, usage, the correction itself)
    "task",          # actually did what the user asked
    "format",        # fully bilingual, clean (no inline pinyin), well-formed
]


def load_eval(n: int | None) -> list[dict]:
    rows = [json.loads(line) for line in c.EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:n] if n else rows


def build_model():
    """Base 7B in 4-bit with the adapter attached (toggled on/off per generation)."""
    tok = AutoTokenizer.from_pretrained(c.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=c.TRAIN.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=c.TRAIN.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, c.TRAIN.bnb_4bit_compute_dtype),
    )
    base = AutoModelForCausalLM.from_pretrained(
        c.BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=getattr(torch, c.TRAIN.bnb_4bit_compute_dtype),
    )
    model = PeftModel.from_pretrained(base, str(c.OUTPUT_DIR))  # tuned; disable_adapter() → base
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, messages: list[dict], max_new_tokens: int = 512) -> str:
    """Greedy generation from the system+user messages (drops the gold assistant turn)."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_pairs(model, tok, rows: list[dict]) -> list[dict]:
    results = []
    for i, row in enumerate(rows, 1):
        ctx = row["messages"][:-1]                 # system + user
        gold = row["messages"][-1]["content"]      # reference tutor answer
        with model.disable_adapter():
            base_out = generate(model, tok, ctx)
        tuned_out = generate(model, tok, ctx)
        results.append({
            "task": row["task"],
            "user": ctx[-1]["content"],
            "gold": gold,
            "base": base_out,
            "tuned": tuned_out,
        })
        print(f"  [{i}/{len(rows)}] {row['task']}")
    return results


def write_report(results: list[dict]) -> None:
    path = c.OUTPUT_DIR / "eval_report.md"
    lines = ["# Before/after eval — base vs. QLoRA tutor\n"]
    for i, r in enumerate(results, 1):
        lines += [
            f"## {i}. `{r['task']}`",
            f"**User:** {r['user']}\n",
            f"**Base (before):**\n\n> {r['base'].replace(chr(10), chr(10) + '> ')}\n",
            f"**Tuned (after):**\n\n> {r['tuned'].replace(chr(10), chr(10) + '> ')}\n",
            "---\n",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {path}")


# --------------------------------------------------------------------------- #
# Optional Claude-as-judge
# --------------------------------------------------------------------------- #
def judge(results: list[dict]) -> None:
    from anthropic import Anthropic

    client = Anthropic()
    axes = ", ".join(RUBRIC_AXES)
    wins = {"base": 0, "tuned": 0, "tie": 0}
    per_axis = {ax: {"base": 0.0, "tuned": 0.0} for ax in RUBRIC_AXES}

    for i, r in enumerate(results, 1):
        prompt = (
            "You are grading two Chinese-tutor answers to the same student request. "
            f"Score each 1–5 on these axes: {axes}. "
            "level_fit = stays at/below HSK-5 and scaffolds; correctness = the Chinese is right; "
            "task = did what was asked; format = fully bilingual (Chinese + English), clean, no per-character pinyin.\n\n"
            f"STUDENT: {r['user']}\n\nANSWER_A (base):\n{r['base']}\n\nANSWER_B (tuned):\n{r['tuned']}\n\n"
            'Return ONLY JSON: {"A": {axis: score,...}, "B": {axis: score,...}, "winner": "A"|"B"|"tie"}.'
        )
        try:
            resp = client.messages.create(
                model=c.TEACHER_MODEL, max_tokens=400, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception as e:  # noqa: BLE001 — best-effort judge
            print(f"  judge skipped [{i}]: {e}")
            continue
        winner = {"A": "base", "B": "tuned", "tie": "tie"}.get(data.get("winner"), "tie")
        wins[winner] += 1
        for ax in RUBRIC_AXES:
            per_axis[ax]["base"] += float(data["A"].get(ax, 0))
            per_axis[ax]["tuned"] += float(data["B"].get(ax, 0))
        print(f"  judged [{i}/{len(results)}] -> {winner}")

    n = max(1, sum(wins.values()))
    print("\n=== Claude-judge summary ===")
    print(f"wins: base {wins['base']} | tuned {wins['tuned']} | tie {wins['tie']}  (n={sum(wins.values())})")
    print("mean score per axis (base -> tuned):")
    for ax in RUBRIC_AXES:
        print(f"  {ax:12s} {per_axis[ax]['base']/n:.2f} -> {per_axis[ax]['tuned']/n:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Before/after eval for the HSK-5 tutor.")
    ap.add_argument("--n", type=int, default=20, help="how many held-out prompts to eval")
    ap.add_argument("--judge", action="store_true", help="also score with Claude (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    rows = load_eval(args.n)
    print(f"evaluating on {len(rows)} held-out prompts...")
    model, tok = build_model()
    results = run_pairs(model, tok, rows)
    write_report(results)
    if args.judge:
        judge(results)


if __name__ == "__main__":
    main()
