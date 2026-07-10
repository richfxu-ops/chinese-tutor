# CLAUDE.md

> Coding preferences for this repo. Project facts, plans, and tasks live in `planning/` — see the index at the bottom. Keep this file prefs-only and lean; it's read into context every session.

## How to work with me

- **Permission before code.** Don't write or edit code until I've approved the plan — propose the approach first.
- **Plan first.** For any non-trivial change, lay out the approach and the files you'll touch before implementing.
- **Work on a side branch.** Create a branch for the change; don't commit straight to the main branch.
- **Small, modular chunks.** Deliver changes in reviewable pieces, not one giant diff.
- **Answer the question I asked.** Address the actual ask before volunteering extras; don't expand scope unprompted.
- **Ask, don't guess.** If a decision depends on my setup or is ambiguous, ask rather than assume.
- **Match the codebase.** Follow the patterns, style, and conventions already in the repo over any default preferences.
- **Be concise and direct.** Minimal preamble, no filler, no performed enthusiasm; straight talk over hedging.
- **Calibrated honesty.** Say "I don't know" when it's true; flag uncertainty instead of pivoting confidently.
- **Teach the modern stack as you go.** I'm strong in Python / PyTorch / DL research but newer to the fine-tuning + shipping ecosystem (PEFT, TRL/`SFTTrainer`, bitsandbytes, Gradio, HF Hub). Briefly explain the unfamiliar parts and why, idiomatically — don't dumb down the ML.
- **Keep it a vibe project.** Small dataset, light qualitative eval, don't over-engineer. If scope starts creeping, say so and we cut.
- **Long explanations → a local HTML file, opened in my browser.** When an explanation would be long or detailed, write it to a local HTML file under `docs/`, then run `open <file>` (macOS) to launch it in my default browser — the markdown link opens the in-app preview, which I don't want. Leave a short summary + the `open <file>` command in chat.

## Planning docs (where the project detail lives)

- `planning/PLAN.md` — what we're building: purpose, scope, features, roadmap, open questions.
- `planning/ARCHITECTURE.md` — repo structure, module responsibilities, run commands, conventions to match.
- `planning/TASKS.md` — the task board (source of truth for work to do).
- `planning/DECISIONS.md` — dated log of decisions and their rationale.
- `planning/dashboard.html` — visual task tracker that reads `TASKS.md`.

**Read `planning/PLAN.md` and `planning/TASKS.md` at the start of a work session.** When work is completed, update `TASKS.md`; when a notable choice is made, log it in `DECISIONS.md`.
