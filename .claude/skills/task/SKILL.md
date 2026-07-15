---
name: task
description: Pick up a task by name — reads its file in planning/tasks/, checks its status, and tells you what mode to operate in (scope, design, implement, or review)
argument-hint: "<task-name> (e.g. 'auth-flow', with or without .md)"
---

# Task

Pick up and work on a task. The task's current **status** determines what you should do — never skip a stage, and especially never jump from To Do or In Proposal straight into writing implementation code.

**Argument:** `$ARGUMENTS` — the task name (with or without `.md`)

Task files live in `planning/tasks/`, one file per task, kebab-case filenames. The board in `planning/TASKS.md` stays the at-a-glance view; task files are where scoping, design, and progress live. Keep the two in sync (see **Board sync** below).

## Step 1: Find the task file

Look for `planning/tasks/$ARGUMENTS.md` (append `.md` if missing). If no match, list the closest existing task files and ask whether the user meant one of those. If they confirm a new task, switch to **Create**.

## Step 2: Create (no match found)

1. **Pick a size** — `Small`, `Medium`, or `Large` (see Sizes below). Usually obvious from the description; state your pick briefly so the user can override. Only ask when genuinely unclear.
2. **Scaffold the file** (replace `<today>` with today's date, `YYYY-MM-DD`). Keep the title short (2–4 words):
   ```markdown
   ---
   status: To Do
   size: <Small|Medium|Large>
   created: <today>
   title: <Title>
   ---

   ## Context
   ```
3. **Add a board line** to `planning/TASKS.md` under `Backlog` (or `Next` if it's up soon), linking to the file:
   `- [ ] (P2) [<Title>](tasks/<name>.md) #<tag>`
4. Drop into the **To Do** flow and start scoping.

Not every board line needs a task file — trivial one-liners (typo fix, dep bump) can live on the board alone. Create a task file when the work involves real scoping, design decisions, or multi-step implementation.

## Step 3: Read the task file

Read the whole file. Note `status` and `size` in the frontmatter, checklist state (`- [ ]` / `- [x]`), and any design context or constraints. Announce which task you're picking up, its status, and its size.

## Step 4: Operate based on status

### To Do
Noted but not scoped. Your job is to **help define it**.
- Discuss scope and approach with the user
- Draft an initial approach and plan checklist in the task file
- Move status to **In Proposal** when the user is ready to iterate on the design

### In Proposal
The user is actively iterating on the design. Your job is to **discuss and refine**.
- Talk through decisions; record them in the task file with rationale, not just conclusions
- When a decision changes, **edit the existing section in place** — don't append a new draft next to the old one (see Task File Shape)
- **Do NOT write implementation code** — only the task file and planning docs
- If scope grows meaningfully, re-check whether the declared size still fits and flag mismatches

**Before moving to In Progress:** run **`/task-tidy`** and address findings, and move on only when the user explicitly signs off on the design. If the task is Large, strongly suggest splitting it into multiple task files first (accept "keep as one" without argument).

### In Progress
Design signed off. Your job is to **implement**.

> **Hard rule — check items off as you go.** After completing any checklist item, your *very next action* is ticking it off in the task file (`- [ ]` → `- [x]`). The task file is the handoff document; an unchecked box signals incomplete work.

- Work on a **side branch**, never directly on main; tell the user which branch
- If the design shifts during implementation, **update the task file's Approach/Decisions in place** — it stays the source of truth. Flag design problems to the user instead of making unilateral design changes
- Move status to **In Review** when all checklist items are done

### In Review
Implementation done. Your job is to **verify**.
- Run the repo's check commands (see `planning/ARCHITECTURE.md`) and review the implementation for correctness — depth per size (see Sizes)
- **Share findings in chat first** so the user can push back before anything is finalized
- Run **`/task-tidy`** on the task file
- **Doc review:** promote finalized decisions into `planning/DECISIONS.md` (dated, with rationale) and update `planning/PLAN.md` / `planning/ARCHITECTURE.md` if the work changed the plan or the structure. If nothing needs updating, say "doc review: nothing to update" explicitly
- Wait for the user to confirm before merging
- Move status to **Complete** when merged

### Complete
Nothing to do — tell the user it's already complete. When flipping a task to Complete, stamp `completed: <today>` in the frontmatter.

## Board sync (planning/TASKS.md)

On every status change, update the task's line on the board:

| Task status | Board column | Extras |
|---|---|---|
| To Do | Backlog or Next | — |
| In Proposal | In Progress | add `#proposal` tag |
| In Progress | In Progress | drop `#proposal` |
| In Review | In Progress | add `#review` tag |
| Complete | Done | check the box `[x]`, drop `#review` |

Keep the markdown link to the task file on the line — the dashboard renders it as a clickable card.

## Sizes

Sized by **decision surface, not file count** — a 100-file rename is Small; a 3-file new subsystem with hard tradeoffs is Medium+.

- **Small** — mechanical or well-understood; no real design decisions. *Proposal:* a quick sanity-check, accept "yeah, go" readily. *Review:* the obvious checks only. Often board-line-only, no task file.
- **Medium** — a handful of decisions or a few subsystems; the default for most tasks. *Proposal:* real design discussion with tradeoffs. *Review:* full path — checks, tests, and manual verification where relevant.
- **Large** — many parts *and* significant complexity; rare by design. Prefer splitting into multiple task files before implementation. If kept as one, break the Plan into numbered **Parts**, each with its own branch/PR, reviewed before the next Part starts; status stays In Progress until all Parts are done.

## Task File Shape

Order sections so a top-down skim gives the big picture first:

1. **Context** — why the task exists. 1–3 paragraphs.
2. **Approach** — the chosen shape in prose (a Small can collapse this into Context).
3. **Decisions** — load-bearing tradeoffs with rationale, one bullet each. Skip if Approach covered it.
4. **Technical detail** *(optional)* — schemas, interfaces, examples. After the conceptual sections, not before.
5. **Plan** — flat checklist of implementation work. Rationale lives in Approach/Decisions, not packed into Plan items.

Writing rules:
- **Prune, don't append.** When a decision shifts, edit the section in place. A short "considered X, went with Y because Z" is fine; coexisting drafts are not.
- **No recap sections** restating what Approach/Decisions already say.
- **Bullets stay bullets.** A Plan item needing a paragraph of rationale means that paragraph belongs in Approach/Decisions.
- Capture decisions a future reader would revisit — not every micro-choice.

## Tidy check (`/task-tidy`)

The companion `/task-tidy` skill (`.claude/skills/task-tidy/SKILL.md`) audits a task file against the Task File Shape rubric above — stacked revisions, stale checkboxes, status/size mismatches, recap bloat, board-line drift. It's the cued action at the two gate transitions (In Proposal → In Progress, In Review → Complete), and can be invoked any time the file seems to have drifted. It reports findings in chat; address them before transitioning.

## Rules

- Always announce the task, its status, and its size when you pick it up
- Never skip statuses; never write implementation code before In Progress
- Update frontmatter status and the board line on every transition
- **The task file is the primary handoff document.** Any future session may pick up the next phase — keep the file current with state, completed work, and open questions; don't rely on chat context carrying over
