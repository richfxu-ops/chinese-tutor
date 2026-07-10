# Tasks — HSK-5 Mandarin Tutor

> Source of truth for work. Format is fixed so `dashboard.html` can parse it:
> columns are `## ` headings; tasks are `- [ ]` / `- [x]`; optional `(P0)`–`(P3)` priority and `#tags`.

## Backlog
- [ ] (P1) train.py — QLoRA SFT (4-bit base + PEFT LoRA + TRL SFTTrainer), save adapter #train #M2
- [ ] (P2) Colab notebook wrapper to run train.py on GPU #train #M2
- [ ] (P1) eval.py — before/after on held-out prompts + rubric scaffold #eval #M2
- [ ] (P2) Merge adapter into base model #ship #M3
- [ ] (P1) app.py — Gradio ChatInterface over merged model #ship #M3
- [ ] (P2) README with before/after examples #ship #M3
- [ ] (P3) Optional: deploy to HF Spaces #ship #M3
- [ ] (P3) Decide: keep 1.5B or bump to 3B (after M2 eval) #train

## Next
- [ ] (P1) config.py — single source of truth (model ids, paths, hyperparams, tasks, system prompt) #data #M1
- [ ] (P1) hsk5_vocab.txt + hsk5_grammar.txt — curated HSK-5 seed #data #M1
- [ ] (P1) gen_data.py — Claude teacher → train.jsonl / eval.jsonl #data #M1
- [ ] (P2) requirements.txt + .gitignore (data/, outputs/, keys) #infra #M1

## In Progress
- [ ] Scaffold repo (CLAUDE.md + planning/ + dashboard) #infra

## Done
- [x] Initialize repository + feature branch (feat/hsk5-tutor)
- [x] Plan approved: QLoRA-on-Colab, Qwen2.5-1.5B-Instruct, Claude teacher, ~800–1k pairs
