# Tasks — HSK-5 Mandarin Tutor

> Source of truth for work. Format is fixed so `dashboard.html` can parse it:
> columns are `## ` headings; tasks are `- [ ]` / `- [x]`; optional `(P0)`–`(P3)` priority and `#tags`.

## Backlog
- [ ] (P3) ~~Optional: deploy to HF Spaces~~ — not feasible for a 7B on free Spaces; local demo only #ship #M3
- [ ] (P3) v2 roadmap: curriculum coach — Claude writes curriculum.md/progress.md, app.py injects into Qwen prompt (app-layer, no retrain) #roadmap
- [ ] (P3) v2 roadmap: text-to-speech + voice chat (ASR) #roadmap

## Next
- [ ] (P1) RUN: train_colab.ipynb on Colab → adapter → eval → merge → GGUF #train #run
- [ ] (P1) RUN: python app.py locally + verify the demo end-to-end #ship #run
- [ ] (P2) Fill README before/after from outputs/eval_report.md #ship #M3

## In Progress

## Done
- [x] Initialize repository + feature branch (feat/hsk5-tutor)
- [x] Plan approved: QLoRA-on-Colab, Qwen2.5-1.5B-Instruct, Claude teacher, ~800–1k pairs
- [x] Scaffold repo (CLAUDE.md + planning/ + dashboard) + GitHub remote #infra
- [x] config.py — single source of truth (models, paths, tasks, system prompt, TrainConfig) #data #M1
- [x] hsk5_vocab.txt (161) + hsk5_grammar.txt (42) — curated HSK-5 seed #data #M1
- [x] requirements.txt (local) + requirements-train.txt (Colab) + .gitignore #infra #M1
- [x] Product spec v1 locked (D1–D7): bilingual + app-side reading layer #data #M1
- [x] gen_data.py — Claude teacher → train.jsonl / eval.jsonl (dry-run verified) #data #M1
- [x] Base model → Qwen2.5-7B-Instruct #train #M2
- [x] train.py — QLoRA SFT (4-bit base + LoRA + TRL SFTTrainer), save adapter #train #M2
- [x] train_colab.ipynb — GPU notebook wrapper (upload → train → download adapter) #train #M2
- [x] eval.py — base vs tuned before/after + optional Claude-judge rubric #eval #M2
- [x] annotate.py — reading layer: pypinyin ruby + jieba/CC-CEDICT hover gloss (tested) #ship #M3
- [x] get_cedict.py — fetch CC-CEDICT (124k entries) + README attribution #ship #M3
- [x] merge.py + notebook GGUF cells — adapter → fp16 → Q4_K_M #ship #M3
- [x] app.py — Gradio chat over quantized 7B with the reading layer #ship #M3
- [x] README with quickstart + before/after template #ship #M3
- [x] Decision: base model = Qwen2.5-7B (resolved, not 1.5B/3B) #train
- [x] RUN: gen_data.py → 810 train / 90 eval, all quality checks clean #data #run
