# CLAUDE.md

> Coding preferences for this repo. Project facts, plans, code, and tooling are indexed in `INDEX.md` (repo root) — read that to find any file. Keep this file prefs-only and lean; it's read into context every session, so it points at the detail rather than holding it.

## How to work with me

- **Permission before code.** Don't write or edit code until I've approved the plan — propose the approach first.
- **Plan first.** For any non-trivial change, lay out the approach and the files you'll touch before implementing.
- **Work on a side branch.** Create a branch for the change; don't commit straight to the main branch.
- **Small, modular chunks.** Deliver changes in reviewable pieces, not one giant diff.
- **Answer the question I asked.** Address the actual ask before volunteering extras; don't expand scope unprompted.
- **Ask, don't guess.** If a decision depends on my setup or is ambiguous, ask rather than assume.
- **Match the codebase.** Follow the patterns, style, and conventions already in the repo over any default preferences.
- **Prioritize maintainability and readability.** Write code the next reader can follow: clear names, small focused functions, comments only where the code can't speak for itself. Prefer the simple, boring solution over the clever one.
- **Be concise and direct.** Minimal preamble, no filler, no performed enthusiasm; straight talk over hedging.
- **Calibrated honesty.** Say "I don't know" when it's true; flag uncertainty instead of pivoting confidently.
- **Teach the modern stack as you go.** I'm strong in Python / PyTorch / DL research but newer to the fine-tuning + shipping ecosystem (PEFT, TRL/`SFTTrainer`, bitsandbytes, Gradio, HF Hub). Briefly explain the unfamiliar parts and why, idiomatically — don't dumb down the ML.
- **Keep it a vibe project.** Small dataset, light qualitative eval, don't over-engineer. If scope starts creeping, say so and we cut.
- **Long explanations → a local HTML file, opened in my browser.** When an explanation would be long or detailed, write it to a local HTML file under `docs/`, then run `open <file>` (macOS) to launch it in my default browser — the markdown link opens the in-app preview, which I don't want. Leave a short summary + the `open <file>` command in chat.

## Where to find things

**`INDEX.md` (repo root) is the table of contents** — where every doc, module, and tool lives, each referenced by path. When you need to find something (a planning doc, a code module, content, a script), read `INDEX.md` first and follow the pointer instead of searching blind. It's kept out of this file on purpose so `CLAUDE.md` stays lean. `planning/ARCHITECTURE.md` is the fuller structural reference.

**Keep `INDEX.md` current.** Any structural change to the repo belongs in `INDEX.md` in the same change — adding an important file, module, or doc; moving, renaming, or removing one; or reorganizing the layout. Add, update, or delete the affected pointer(s) so the map never lies — a stale index is worse than none.

**Read `planning/PLAN.md` and `planning/TASKS.md` at the start of a work session.** When work is completed, update `TASKS.md`; when a notable choice is made, log it in `DECISIONS.md`.

**Task lifecycle.** Substantial tasks are worked via the `/task` skill (`.claude/skills/task/SKILL.md`): statuses flow To Do → In Proposal → In Progress → In Review → Complete, and no implementation code is written before the design is signed off (In Progress). Even without invoking the skill, follow that lifecycle — never skip from an unscoped task straight into code.
