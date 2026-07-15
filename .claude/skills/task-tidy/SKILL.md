---
name: task-tidy
description: Audit a task file for coherence, hygiene, and simplicity — structure, checkboxes, status/metadata, verbosity, board sync. Reports findings in chat; doesn't auto-edit.
argument-hint: "<task-name> (optional — inferred from context if omitted)"
---

# Task Tidy

Audit a task file in `planning/tasks/` against the rubric in the `/task` skill's `## Task File Shape` section. Report findings in chat as a short structured review. Don't auto-edit — the user decides what to fix.

Use at state transitions (In Proposal → In Progress, In Review → Complete) or any time the file seems to have drifted, accumulated stacked revisions, or grown beyond its size.

## Step 1: Identify the target task file

1. **If the user passed an argument** — use it. Resolve to `planning/tasks/<name>.md` the same way `/task` does.
2. **If no argument** — infer from current conversation context. The user is usually mid-discussion about a task when invoking this; that's the target.
3. **If context is unclear** — check whether the current branch name maps to a task file (strip the `feat/`/`fix/`/`docs/` prefix and look for a matching `planning/tasks/<rest>.md`).
4. **If still unclear** — list plausible candidates and ask the user directly. Don't heuristic-guess.

**Always announce the target before auditing** so a misfire is caught immediately:

> Auditing `planning/tasks/auth-flow.md` (inferred from our discussion above).

If the user sees the wrong file, they can re-invoke with an explicit arg.

## Step 2: Audit against the rubric

Load `/task`'s `## Task File Shape` section (`.claude/skills/task/SKILL.md`) — it's the rubric; the categories below reference its outline and writing rules by name. Read the task file end-to-end, then walk each category. Empty categories are fine — don't pad.

### Shape & coherence
Walk the recommended outline (Context → Approach → Decisions → Technical detail → Plan) and the writing rules "Prune, don't append" and "No recap sections" — flag any violations with a location.

### Checkboxes
- Do ticked items match what the body says was done?
- Any unchecked items that clearly look done from other prose or from code changes?
- Any vague items that shouldn't be checkboxes at all?

### Status & metadata
- Does `status` match the file state? Complete requires `completed:`; In Review implies all boxes ticked; In Progress should show visible progress.
- Title reasonable length (2–4 words)?
- Size seems to match content volume — not a Small at 300 lines, not a Large at 30?

### Verbosity & granularity
Walk the writing rules "Bullets stay bullets" and "capture decisions a future reader would revisit, not every micro-choice" — flag violations with a location.

### Board sync
- Does the task's line in `planning/TASKS.md` exist, link to this file, and sit in the column its status maps to (see the `/task` skill's Board sync table)?
- Are the `#proposal`/`#review` tags and the checkbox state consistent with the frontmatter status?

### General hygiene
- Refs to planning docs or other files that don't exist?
- Stale TBD/TODO markers from earlier drafts?
- Code blocks missing language tags?

## Step 3: Report findings

Output a short structured review in chat. Bullet points grouped by the categories above; omit empty categories. Each finding is one line — the specific issue with a location or quoted snippet so the user can see exactly what you mean.

If the file is clean: output one line — **"Looks clean."** — and stop.

Offer to fix specific items on request ("Want me to consolidate the two Decisions blocks, or leave them for you?"). Never auto-edit.

## Rules

- Announce the target file before auditing
- The `/task` skill's `## Task File Shape` section is the rubric — don't invent new criteria
- Report in chat; never silently edit the file
- Keep the review short — omit empty categories
- If nothing's wrong, say so in one line and stop
