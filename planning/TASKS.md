# Tasks — HSK-5 Mandarin Tutor

> Source of truth for work. Format is fixed so `dashboard.html` can parse it:
> columns are `## ` headings; tasks are `- [ ]` / `- [x]`; optional `(P0)`–`(P3)` priority and `#tags`.

## Backlog
- [ ] (P3) ~~Optional: deploy to HF Spaces~~ — not feasible for a 7B on free Spaces; local demo only #ship #M3
- [ ] (P3) v2 roadmap: curriculum coach — Claude writes curriculum.md/progress.md, app.py injects into Qwen prompt (app-layer, no retrain) #roadmap
- [ ] (P3) v2 roadmap: auto-extract vocab from a conversation (jieba ∩ CC-CEDICT) into the deck #roadmap
- [ ] (P3) v2 roadmap: voice chat — speak your Chinese, Whisper-class ASR #roadmap
- [ ] (P3) Wrong-rule follow-up (optional): corrected sentences are reliable but ~5/8 rule EXPLANATIONS are wrong (见面/结婚 transitivity backwards, self-contradiction). Options: shorten/soften rule text in correction data, or targeted rule-accuracy data iteration #data
- [ ] (P3) Watch 14B serve latency (3.9s mean × up-to-3 calls/turn); escape hatch = 14B for main gen, 7B for auxiliary (disambiguation/fills) #ship
- [ ] (P3) Serve: correction chip fires on false corrections (overlap gate can't tell a fabricated fix of a correct sentence from a real one) — mitigate or gate on a "was there actually an error" signal; also no chip in Q&A 修改后 format #ship
- [ ] (P3) Optional: collapse the SYSTEM_PROMPT / SYSTEM_PROMPT_APP train-serve split (deferred from the redesign to keep the retrain's variables focused — DECISIONS 2026-07-11) #data
- [ ] (P3) Optional: stratified correct-sentence eval probes in eval.py (they already flow into eval.jsonl via the split) #eval
- [ ] (P3) Data iteration: enforce English-explanation compliance in conversation corrections (~1/5 slip through, model reproduces it) #data

## Next
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
- [x] Q&A translation fill: Chinese lines the model left untranslated get a local translate pass (one extra call only when gaps exist); fills stored on the message, never fed back to the model; styled identically to authored text #ship
- [x] English prompts get their Chinese under the student's bubble — annotated (ruby + glosses + collectible), both modes #ship
- [x] Word list → click a word to open its card (back view) in the flashcards tab; parent→iframe show-card message, retried for lazy tab mounts #ship
- [x] start-tutor.command launcher + Desktop app bundle with 文 seal icon #ship
- [x] Review-through-conversation (复习模式 checkbox: targets = due cards only) + per-line TTS in chat #ship
- [x] File-backed deck (data/deck.json mirror + fresh-browser restore) #ship
- [x] Streaming replies (plain live bubble → 加注中 → annotated swap-in; scroll-follow) #ship
- [x] 写 stroke-order practice on flashcards (hanzi-writer local lib + get_strokes.py data, quiz overlay) #ship
- [x] LAN=1 option (phone/iPad as touchscreen) + review-checkbox layout fix #ship
- [x] Code review round 3 → 8 fixes: model lock (crash-class), honest stream failures, card-fill line selection, live review-mode toggle, deck union-merge, render_chat(unclosed), fill key, DECISIONS entries #quality
- [x] 聊天 mode Chinese-in/Chinese-out: English prompts translated BEFORE the model sees them; translation-drift stripping on replies #ship
- [x] Starter chips: validity filter (no "How do I say [中文] in Chinese?"), ⟲ refresh button, JSON-array salvage for wrong-closing-bracket rolls #ship
- [x] gen_data.py + config.py redesign (2026-07-11): correct_sentence response-type modes (error .40 / correct .35 / polish .20 / ambiguous .05), rule-consistency instruction, error-free conversations (25%), target-word invitation bias, BASE_MODEL → Qwen2.5-14B. Dry-run + offline record-path tests passed. #data #train
- [x] Live smoke test (2026-07-11): all 4 correction modes verified against the teacher; fixed GEN_MAX_TOKENS 4096→8192 (ambiguous replies truncated + dropped otherwise) #data #run
- [x] RUN `python gen_data.py` (2026-07-12): regenerated 900 train / 100 eval with the new mode mix. 280/280 batches, ZERO drops. correct_sentence modes error 70 / correct 65 / polish 35 / ambiguous 10 (matches 40/35/20/5 target); modes present in both splits; content spot-checks clean (no invented errors). Old data backed up to data/*.pre-14b-bak. #data #run
- [x] Retrain on Colab A100 (2026-07-12): 14B QLoRA via the disk-managed notebook (sanity run → full train → eval → merge → GGUF, intermediates deleted as consumed) → ~9GB Q4_K_M into outputs/, 7B kept as hsk5-tutor-q4_k_m.7b-v2.gguf rollback #train #run
- [x] Re-run qa_harness against the 14B (2026-07-12): false corrections ~55%→~0%, recall 8/8, fixes 8/8, driving 36/37; contracts intact; ~2.2× latency. Wrong-RULE issue persists (~5/8, capability-limited). See DECISIONS 2026-07-12. #eval
- [x] Reading practice tab (2026-07-12): model writes a passage + 3 comprehension questions on a topic (blank = random CONV_TOPIC), rendered through the reading layer (pinyin/hover/click-to-collect), English translation shown, per-passage 🔊, reveal-answer buttons. Verified live end-to-end; later made level-aware + doubled (see HSK 4/6 entry). #ship
- [x] Chinese→English fill (2026-07-13): a Chinese message (≥4 hanzi, both modes) gets its English translation beneath the bubble — the mirror of the English→Chinese fill; question-shaped messages are translated, not answered, and a preamble-stripping regex removes "Sure, here's the translation:"-style lead-ins. #ship
- [x] Shareable setup (2026-07-13): setup.sh (venv + requirements-app.txt serve-only deps + cedict/strokes fetch + resumable ~9GB model download from HF with a truncation check) + README "just run it" quickstart. GGUF uploaded to richfxu/hsk5-tutor-14b-gguf (public; anonymous download verified, 8,988,110,144 bytes); repo made public after a history secret-scan. #ship
- [x] Neural TTS (2026-07-13): 🔊 buttons request a Microsoft zh-CN neural clip (edge-tts, free, no key) via a hidden channel; MP3 returned as a base64 data URI, cached per (voice, line); falls back to browser speechSynthesis on failure/offline. Verified end-to-end. #ship
- [x] TTS controls (2026-07-13): 女声/男声 voice picker + 0.5–1.5× playback-speed slider (pitch preserved, live-adjust); 复习模式 checkbox restyled transparent. #ship
- [x] Flashcards neural TTS (2026-07-13): card 🔊/autoplay proxied to the parent's neural channel via postMessage (voice + speed respected), with its own persisted 女声/男声 toggle; browser voice standalone/fallback. Verified cross-frame end-to-end. #ship
- [x] Maintainability review + cleanup (2026-07-13): CLAUDE.md gains the maintainability/readability preference; whole codebase audited against it (report: docs/maintainability-review-2026-07-13.html — verdict: meets the standard, no dead code). Fixed the 8 stale/lying comments (7B/T4 leftovers, the false "keep prompts identical" invariant, flashcards schedule() "mutates a copy") + 5 small cleanups (SERVER_STARTERS rename, afterDeckChange() helper, MIN_GGUF_BYTES, regex hoist, gold shown in eval report). #quality
- [x] Tier-3 structural polish (2026-07-13): Entry NamedTuple (annotate), print_dry_run + _call_teacher retry dedup + MIN_CONV_TURNS (gen_data), judge reuses gen_data parsing helpers lazily (eval), msg_is_english/_chinese rename + _fill_under_last_line + named 0.6 thresholds (app.py), renderWordlist → template literals (byte-identity proven) + escHtml + TTS_FALLBACK_MS twins (app.js/flashcards), .spk base style + .b/.t/.u decode comment (app.css). annotate + gen_data dry-run proven byte-identical. #quality
- [x] Translators execute-instead-of-translate fix (2026-07-13, user-reported): 用“珍惜”写三个例句 got ANSWERED in English under the bubble — the guard only covered questions, not imperatives. Both translator prompts now cover requests/instructions explicitly, plus a length-expansion backstop drops any 'translation' several times longer than its source (no fill beats a wrong fill; passage-length inputs can't trip it). Verified live: 8/8 cases across both directions. #ship
- [x] Reading + card reliability fixes (2026-07-13, user-reported): passage builds from 4 simple calls (halves as plain text, questions as a salvageable JSON array, translation from the FINAL text at temp 0 — the do-everything JSON call mangled its structure and the co-generated translation drifted); card examples accept 离合词 split forms (打了三年交道 was rejected by `word in line`, discarding perfect output) and example_en falls back to the temp-0 translator. Verified live: 239-hanzi passage w/ faithful translation; 15/16→ the failing word now passes 3/3 rolls. #ship
- [x] HSK 4/6 levels + longer passages (2026-07-13): level selector (prompt-steered; level 5 byte-identical to trained prompts), per-level seed lists, level-aware starters/targets/passages; passages doubled (~100→~200 hanzi) via a two-call generate-then-continue (one call snaps to the trained reply length — measured 87–153 across four prompt variants). #ship
