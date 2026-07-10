# Decisions — HSK-5 Mandarin Tutor

> Dated log of notable choices and *why*, so rationale isn't lost. Newest first.

## 2026-07-09 — Product spec v1 locked (D1–D7)
- **Decision:** Fully **bilingual** output (D1). Pinyin + per-word English gloss are an **app-side reading layer**, not model output (D2): `pypinyin` renders pinyin ruby over every character, `jieba` segments words, bundled **CC-CEDICT** supplies hover glosses, rendered as HTML `<ruby>` + `title` tooltips. Single-turn training data (D3). No quiz mode in v1 (D4). Above-level requests simplified + flagged (D5). Simplified characters only (D6). Gradio chat + starter buttons (D7). **Voice chat + TTS deferred to v2 roadmap.**
- **Why:** A 1.5B model shouldn't generate per-character pinyin (verbose, error-prone) — a library is always correct and needs zero training data. Bilingual is the only content change; the reading layer is pure rendering. Keeps v1 small while delivering the headline "readable" feature.
- **Implications:** `SYSTEM_PROMPT` + task instructions go bilingual and drop inline pinyin (done). New `annotate.py` module + CC-CEDICT asset (~4MB, CC-BY-SA → README attribution). `app.py` renders HTML (ruby + title). Hover tooltips are desktop-only (no mobile hover) — acceptable for a demo.

## 2026-07-09 — Base model: Qwen2.5-1.5B-Instruct
- **Decision:** Start with `Qwen/Qwen2.5-1.5B-Instruct`, not 3B.
- **Why:** Strong Chinese for its size; fast train + fast Gradio inference keeps the vibe-project loop tight. 3B is an easy one-line bump if M2 eval shows quality is short.
- **Implications:** Fast iteration; revisit model size after seeing eval results.

## 2026-07-09 — Training path: QLoRA on Colab (not local MLX)
- **Decision:** Train via HF `peft` + `trl` `SFTTrainer` + `bitsandbytes` 4-bit QLoRA on **paid Colab**; keep data-gen and the app local on the Mac.
- **Why:** This is the transferable, industry-standard fine-tuning stack (the user's stated learning goal) and the most reproducible/shareable path; it maps cleanly to HF Hub + Spaces shipping. MLX is excellent and fully local on the M5, but Apple-specific and doesn't map to the HF ecosystem. bitsandbytes 4-bit is CUDA-only, so training is a GPU/Colab step.
- **Implications:** `train.py` runs on Colab; `app.py` runs the merged model locally on MPS (no bitsandbytes at inference).

## 2026-07-09 — Data: synthetic, Claude teacher, ~800–1,000 pairs
- **Decision:** Generate ~800–1,000 chat-format pairs with Claude (Anthropic API), seeded from a curated HSK-5 vocab/grammar list, ~10% held out for eval. Six task types.
- **Why:** Enough signal for a LoRA on a small model, cheap + fast to generate in one session, and level-controllable via the seed list + system prompt. Matches "keep it small / don't chase eval perfection."
- **Implications:** Level control depends on seed quality + prompt constraints; start with a curated in-repo seed and expand only if variety is thin.
