# Index — HSK-5 Mandarin Tutor

> The map of the repo: where every doc, module, and tool lives. `CLAUDE.md` points here so it can stay lean (prefs only). **Start here to find the right file, then open that file for the detail** — don't expect the depth to live in this list. `planning/ARCHITECTURE.md` is the fuller structural reference.
>
> Keep this current: reflect any structural change here in the same change — a file, module, or doc added, moved, renamed, or removed, or the layout reorganized. Add, update, or drop the affected pointer(s). A stale index is worse than none.

## Planning & project docs

- `planning/PLAN.md` — what we're building: purpose, scope, features, roadmap, open questions.
- `planning/ARCHITECTURE.md` — repo structure, module responsibilities, build/test/run commands, conventions to match. The deep version of this map.
- `planning/TASKS.md` — the task board (source of truth for work to do).
- `planning/tasks/` — one file per substantial task: its status, size, design, and checklist.
- `planning/DECISIONS.md` — dated log of decisions and their rationale.
- `planning/dashboard.html` — visual task tracker that reads `TASKS.md`.

## Pipeline (data → train → serve)

- `config.py` — single source of truth: model ids, paths, HSK level, data-gen + training hyperparams. Imported by everything below.
- `gen_data.py` — calls Claude to synthesize tutor training pairs → `data/train.jsonl`, `data/eval.jsonl`.
- `train.py` — QLoRA SFT; produces an adapter under `outputs/`. Colab-runnable (see the notebook).
- `train_colab.ipynb` — thin notebook wrapper to run `train.py` on a Colab GPU.
- `merge.py` — merges the trained LoRA adapter into the base model for local serving.
- `eval.py` — before/after generation on held-out prompts + rubric scaffold.
- `annotate.py` — reading layer: CJK text → `<ruby>` pinyin + per-word hover glosses (HTML). Used by `app.py`.
- `app.py` — the tutor app: chat UI over the merged model, rendered through `annotate.py`. Main entry point.

## Reading-layer data & helpers

- `get_cedict.py` — downloads/prepares the CC-CEDICT dictionary. `cedict_ts.u8` is the bundled dict file.
- `get_strokes.py` — fetches stroke-order data → `data/strokes/`.
- `hsk4_vocab.txt` · `hsk4_grammar.txt` · `hsk5_vocab.txt` · `hsk5_grammar.txt` · `hsk6_vocab.txt` · `hsk6_grammar.txt` — seed vocab + grammar points the generator draws from, by HSK level.

## Frontend (`web/`)

- `web/flashcards.html` · `web/header.html` — HTML views served by the app.
- `web/app.js` · `web/app.css` — client-side behavior and styling.
- `web/vendor/` — bundled third-party frontend libs.

## Config, setup & run

- `requirements.txt` · `requirements-app.txt` · `requirements-train.txt` — Python deps (base / serving app / training).
- `setup.sh` — environment setup script.
- `start-tutor.command` — double-clickable macOS launcher for the tutor app.
- `.claude/launch.json` — dev-server launch config for the in-app browser preview.
- `.gitignore` — `data/` and `outputs/` artifacts are git-ignored; regenerate rather than commit.

## Generated artifacts (git-ignored)

- `data/` — generated training/eval JSONL, `deck.json`, `strokes/`.
- `outputs/` — training adapters / merged-model checkpoints.
- `llama.cpp/` — vendored llama.cpp for local GGUF quantized serving on Apple Silicon.

## Docs & explanations (`docs/`)

- `docs/index.html` — landing page linking the explanation docs below.
- `docs/how-llms-work.html` · `docs/codebase-walkthrough.html` · `docs/learner-memory.html` — conceptual explainers.
- `docs/colab-guide.html` · `docs/training-run.html` · `docs/finish-on-mac.html` — walkthroughs for the training + serving workflow.
- `docs/qa-report-2026-07-11.html` · `docs/maintainability-review-2026-07-13.html` — point-in-time review reports.
- `README.md` — project overview and quickstart.

## Agent tooling (`.claude/skills/`)

- `.claude/skills/task/SKILL.md` — the `/task` lifecycle skill (To Do → In Proposal → In Progress → In Review → Complete).
- `.claude/skills/task-tidy/SKILL.md` — the `/task-tidy` audit skill, run at the lifecycle's gate transitions.

If deeper detail is needed, **follow the path and read the file** — don't duplicate it here.
