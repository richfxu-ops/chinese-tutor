# Tasks — HSK-5 Mandarin Tutor

> Source of truth for work. Format is fixed so `dashboard.html` can parse it:
> columns are `## ` headings; tasks are `- [ ]` / `- [x]`; optional `(P0)`–`(P3)` priority and `#tags`.

## Backlog
- [ ] (P1) train.py — QLoRA SFT (4-bit base + PEFT LoRA + TRL SFTTrainer), save adapter #train #M2
- [ ] (P2) Colab notebook wrapper to run train.py on GPU #train #M2
- [ ] (P1) eval.py — before/after on held-out prompts + rubric scaffold #eval #M2
- [ ] (P1) annotate.py — reading layer: pypinyin ruby + jieba/CC-CEDICT hover gloss → HTML #ship #M3
- [ ] (P2) Bundle CC-CEDICT + README attribution (CC-BY-SA) #ship #M3
- [ ] (P2) Merge adapter into base model #ship #M3
- [ ] (P1) app.py — Gradio chat (HTML render: ruby + title) over merged model #ship #M3
- [ ] (P2) README with before/after examples #ship #M3
- [ ] (P3) Optional: deploy to HF Spaces #ship #M3
- [ ] (P3) Decide: keep 1.5B or bump to 3B (after M2 eval) #train
- [ ] (P3) v2 roadmap: text-to-speech + voice chat (ASR) #roadmap

## Next
- [ ] (P1) Smoke-test + full run of gen_data.py (needs ANTHROPIC_API_KEY) #data #M1

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
