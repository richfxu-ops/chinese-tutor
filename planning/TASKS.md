# Tasks — HSK-5 Mandarin Tutor

> Source of truth for work. Format is fixed so `dashboard.html` can parse it:
> columns are `## ` headings; tasks are `- [ ]` / `- [x]`; optional `(P0)`–`(P3)` priority and `#tags`.

## Backlog
- [ ] (P3) ~~Optional: deploy to HF Spaces~~ — not feasible for a 7B on free Spaces; local demo only #ship #M3
- [ ] (P3) v2 roadmap: curriculum coach — Claude writes curriculum.md/progress.md, app.py injects into Qwen prompt (app-layer, no retrain) #roadmap
- [ ] (P3) v2 roadmap: auto-extract vocab from a conversation (jieba ∩ CC-CEDICT) into the deck #roadmap
- [ ] (P3) Upgrade chat TTS from browser speechSynthesis to edge-tts neural voices #roadmap
- [ ] (P3) v2 roadmap: voice chat — speak your Chinese, Whisper-class ASR #roadmap

## Next
- [ ] (P2) Fill README before/after from outputs/eval_report.md #ship #M3
- [ ] (P3) Data iteration: enforce English-explanation compliance in conversation corrections (~1/5 slip through, model reproduces it) #data

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
- [x] web/flashcards.html — self-contained SM-2 review widget (localStorage, verified) #ship
- [x] web/flashcards.html — 🔊 browser-TTS pronunciation + auto-play toggle #ship
- [x] docs/ reference pages (how-llms-work, training-run) + long-explanation pref #infra
- [x] RUN: python app.py locally + verify the demo end-to-end #ship #run
- [x] Fix reading layer in Gradio: dark-mode-safe colors + CSS hover gloss (verified in-browser) #ship #M3
- [x] Redesign app UI: "teacher's red ink" paper aesthetic (theme + launch css, light-only) #ship #M3
- [x] Click-to-collect: chat word → flashcards deck (localStorage, dedup, toast) #ship #M3
- [x] Flashcards as in-app tab (iframe srcdoc, live storage sync, paper restyle) #ship #M3
- [x] Chat transcript: scrollbar cap + auto-scroll to newest reply #ship #M3
- [x] Chat polish: random starter chips per load, 🔊 browser TTS on replies, follow-up-question prompt rule #ship #M3
- [x] Conversation mode: 聊天 toggle — Chinese-forward chat, corrects errors in passing, ends with a question #ship #M3
- [x] 词表 word-list tab: table of collected cards + per-row removal, synced with flashcards #ship #M3
- [x] Conversation mode v2: tutor drives (topics, self-disclosure, open questions) + target words from the student's deck #ship #M3
- [x] Generate 100 multi-turn conversations (2 smoke iterations: closed-world corrections fix) → 900/100 jsonl #data
- [x] Retrain v2 on Colab A100 (114 steps; T4 attempts hit stale-data + eval-OOM, both fixed) → merge → GGUF → swapped into outputs/, v1 kept as rollback #train
- [x] Verified v2 weights live: drives conversation, pushes past lazy answers, corrects planted 见面她 error, weaves target words; Q&A bilingual format intact #train
- [x] Correction flashcards: tutor corrections get a save chip → kind:'fix' cards (wrong sentence → fix + rule) in the deck, review widget, word list, Anki export #ship
- [x] Voice input: Web Speech API mic button (zh-CN, interim streaming, no auto-submit); feature-detected, graceful permission errors — needs a real-mic check in Chrome #ship
- [x] In-context sense disambiguation: one extra model call/turn picks the CEDICT sense per ambiguous word (地道 → dì dao authentic); overrides fix tooltip + ruby #ship
- [x] Model-written flashcard examples (async via hidden req/res channel) + CJK-ratio fix for the scraped placeholder #ship
- [x] Editable cards: click-to-edit gloss/example in the word list; ✎ edit form in the review widget #ship
- [x] Model-written starter chips at startup (seeded random vocab/grammar, one per task type; static pool kept as loud fallback) #ship
- [x] Code review round 1 (session diff): 6 confirmed findings fixed (false correction chips, gen_card_example guards, startup-safe starters, py<3.12 f-string, task-aware eval rubric, fix-card 'p' TTS) + follow-ups 8–10 (focus-time deck sync, tips on message dicts, helper dedup) #quality
- [x] Flashcard example sentences get English translations (separate example_en field; review card, edit form, word list, Anki export) #ship
- [x] 问老师 button on card backs → auto-asks in the chat tab (word: re-explain; 改错: why) #ship
- [x] Code review round 2: 问老师 deck-sync bypass closed + e.source auth on the message channel, trailing-English trim on examples, word-list translation editing, comment/dedup cleanups; overlap gate + pinyin-leak candidates empirically cleared against training data #quality
- [x] Model fills missing definitions on collect: words CEDICT lacks (一只, 很累…) get a model-written gloss alongside the example via the card channel #ship
- [x] Hover tooltips too: unglossed words get model-written definitions inside the existing per-turn disambiguation call (flat JSON: number=sense pick, string=definition) #ship
- [x] Q&A translation fill: Chinese lines the model left untranslated get a local translate pass (one extra call only when gaps exist); fills render as italic .fill-en lines, stored on the message, never fed back to the model #ship
