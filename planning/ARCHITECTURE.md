# Architecture — HSK-5 Mandarin Tutor

> How the repo is organized and how to work it. Keep commands accurate — mark anything unverified as TODO.

## Stack
- **Language:** Python 3.11+
- **Data gen:** `anthropic` SDK (Claude teacher model), local on the Mac.
- **Training:** QLoRA — `transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes` (4-bit), `datasets`, `accelerate`. Runs on **Colab (CUDA GPU)**; bitsandbytes 4-bit is CUDA-only.
- **Serving/demo:** `gradio` (`ChatInterface`), local on the Mac (MPS/CPU inference of merged model).
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`.

## Repo structure
- `config.py` — single source of truth: model ids, paths, HSK level, data-gen + training hyperparams. Imported by `gen_data.py`, `train.py`, `eval.py`, `app.py`.
- `hsk5_vocab.txt` / `hsk5_grammar.txt` — seed vocab + grammar points the generator draws from.
- `gen_data.py` — calls Claude to synthesize pairs → `data/train.jsonl`, `data/eval.jsonl`.
- `train.py` — QLoRA SFT; produces an adapter under `outputs/`. Colab-runnable (thin notebook wrapper).
- `eval.py` — before/after generation on held-out prompts + a rubric scaffold.
- `app.py` — Gradio chat UI over the merged model.
- `data/`, `outputs/` — generated artifacts (git-ignored).
- `planning/` — project docs + task dashboard.

## Modules / components
- **config → everything:** one `Config` object so gen/train/eval/app stay consistent (same system prompt, tasks, paths).
- **gen_data.py:** for each task type, sample seed vocab/grammar → prompt Claude → parse to the chat schema (`{task, hsk_target, messages:[system,user,assistant]}`) → split train/eval.
- **train.py:** load 4-bit base → attach LoRA (PEFT) → `SFTTrainer` on the chat-formatted JSONL → save adapter.
- **eval.py:** load base and adapter, generate on held-out prompts, print side-by-side + rubric.
- **app.py:** load merged model, wire `gr.ChatInterface` with the tutor system prompt.

## Build / test / run
> Not yet written — updated as files land. Marked TODO until verified.
- Install (local): `pip install -r requirements.txt` — TODO: create requirements.txt
- Generate data: `python gen_data.py` (needs `ANTHROPIC_API_KEY`) — TODO
- Train (Colab): run `train.py` in a GPU notebook — TODO
- Eval: `python eval.py` — TODO
- Run demo: `python app.py` — TODO
- Lint/format: TODO — decide (likely `ruff`), keep light.

## Conventions (match these)
- Small, single-purpose scripts driven by `config.py`; no framework/package layering for a project this size.
- Chat-format data (system/user/assistant messages) so `SFTTrainer`'s chat template applies cleanly.
- Secrets via env var (`ANTHROPIC_API_KEY`); never commit keys or generated data.
- Keep dependencies minimal and idiomatic to the HF ecosystem.

## Gotchas / notes
- **bitsandbytes 4-bit is CUDA-only** → `train.py` is a Colab/GPU step, not local Mac.
- `data/` and `outputs/` are git-ignored; regenerate rather than commit.
- Inference in `app.py` runs the *merged* model on Apple Silicon (MPS) — no bitsandbytes needed there.
