"""Generate synthetic HSK-5 tutoring data with Claude (the teacher model).

For each task in config.TASKS we sample seed word(s)/grammar point(s), ask the
teacher for a batch of examples, parse them into the chat schema, then split
per-task into train/eval jsonl.

Usage:
    python gen_data.py --dry-run        # assemble + print prompts, no API calls (free)
    python gen_data.py --limit 10       # tiny paid smoke test (10 examples/task)
    python gen_data.py                  # full run (needs ANTHROPIC_API_KEY)

Each output record:
    {
      "task": "correct_sentence",
      "hsk_target": ["承担"],
      "messages": [
        {"role": "system",    "content": <SYSTEM_PROMPT>},
        {"role": "user",      "content": <learner query>},
        {"role": "assistant", "content": <ideal tutor reply>}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config as c

# --------------------------------------------------------------------------- #
# Seed loading + sampling
# --------------------------------------------------------------------------- #
def load_seed(path: Path) -> list[str]:
    """Read a seed file, skipping blank lines and '#' comments."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def make_item_batches(items: list[str], n: int, per_call: int, rng: random.Random) -> list[list[str]]:
    """Build `n` target items (shuffled, cycling if n > len) chunked into batches
    of `per_call`. Each item becomes one example, so every example targets a
    distinct seed word/grammar point and coverage is even."""
    if not items:
        raise ValueError("empty seed list — check hsk5_vocab.txt / hsk5_grammar.txt")
    pool: list[str] = []
    while len(pool) < n:
        shuffled = items[:]
        rng.shuffle(shuffled)
        pool.extend(shuffled)
    pool = pool[:n]
    return [pool[i : i + per_call] for i in range(0, n, per_call)]


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def build_teacher_prompt(task: c.TaskSpec, items: list[str]) -> str:
    """The user-turn we send to the teacher. Asks for one example per target
    item, as a JSON array of {user, assistant} objects."""
    kind = "grammar point" if task.needs_grammar else "word"
    targets = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
    return f"""You are creating synthetic training data for a Chinese tutoring model \
aimed at HSK-5 learners.

TASK: {task.instruction}

The tutor (the "assistant" in each example) MUST follow this persona exactly:
---
{c.SYSTEM_PROMPT}
---

Produce exactly {len(items)} examples, ONE for each target {kind} below, in order:
{targets}

For each example:
- "user": a realistic learner request for this task involving the target {kind}. \
Vary phrasing; the learner may write in Chinese, English, or a mix, and may make \
natural mistakes where the task calls for it.
- "assistant": the ideal tutor reply, faithfully following the persona above and \
kept at HSK-5 level.

Return ONLY a JSON array of {len(items)} objects, each {{"user": "...", "assistant": "..."}}. \
No markdown, no commentary."""


# --------------------------------------------------------------------------- #
# Teacher call + parsing
# --------------------------------------------------------------------------- #
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def response_text(resp) -> str:
    """Concatenate the text blocks of a message, skipping thinking blocks.

    Sonnet 5 returns a ThinkingBlock before the TextBlock, so we can't just take
    content[0] — we filter to the text blocks.
    """
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def extract_json_array(text: str) -> list[dict]:
    """Pull the JSON array out of a teacher response, tolerating stray prose or
    ```json fences."""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        raise ValueError(f"no JSON array found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def to_records(task: c.TaskSpec, items: list[str], pairs: list[dict]) -> list[dict]:
    """Wrap teacher {user, assistant} pairs into the final chat schema."""
    if len(pairs) != len(items):
        print(f"  ~ {task.name}: teacher returned {len(pairs)} pairs for {len(items)} items (keeping {min(len(pairs), len(items))})")
    records = []
    for item, pair in zip(items, pairs):
        records.append(
            {
                "task": task.name,
                "hsk_target": [item],
                "messages": [
                    {"role": "system", "content": c.SYSTEM_PROMPT},
                    {"role": "user", "content": pair["user"].strip()},
                    {"role": "assistant", "content": pair["assistant"].strip()},
                ],
            }
        )
    return records


def generate_batch(client, task: c.TaskSpec, items: list[str], retries: int = 2) -> list[dict]:
    """One API call → up to len(items) records. Retries on parse failure."""
    prompt = build_teacher_prompt(task, items)
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.messages.create(
                model=c.TEACHER_MODEL,
                max_tokens=c.GEN_MAX_TOKENS,
                temperature=c.GEN_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            pairs = extract_json_array(response_text(resp))
            return to_records(task, items, pairs)
        except Exception as e:  # noqa: BLE001 — never let one bad batch (API error,
            last_err = e        # rate limit, odd response shape) kill the whole paid run
    print(f"  ! dropped a {task.name} batch after {retries + 1} tries: {last_err}")
    return []


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(limit: int | None, workers: int, dry_run: bool) -> None:
    rng = random.Random(c.TRAIN.seed)
    vocab = load_seed(c.VOCAB_FILE)
    grammar = load_seed(c.GRAMMAR_FILE)

    # Plan the batches for every task up front (cheap, no API).
    plan: list[tuple[c.TaskSpec, list[str]]] = []
    for task in c.TASKS:
        n = min(task.n, limit) if limit else task.n
        seed = grammar if task.needs_grammar else vocab
        for batch in make_item_batches(seed, n, c.EXAMPLES_PER_CALL, rng):
            plan.append((task, batch))

    if dry_run:
        print(f"DRY RUN — {len(plan)} batches planned across {len(c.TASKS)} tasks. "
              f"Showing the first prompt per task:\n")
        seen = set()
        for task, batch in plan:
            if task.name in seen:
                continue
            seen.add(task.name)
            print(f"{'=' * 70}\nTASK: {task.name}  (targets: {batch})\n{'=' * 70}")
            print(build_teacher_prompt(task, batch))
            print()
        return

    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    print(f"Generating {len(plan)} batches with {workers} workers "
          f"(teacher={c.TEACHER_MODEL})...")

    # Fan out the batches; group results back per task for the train/eval split.
    by_task: dict[str, list[dict]] = {t.name: [] for t in c.TASKS}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_batch, client, task, batch) for task, batch in plan]
        for i, fut in enumerate(futures, 1):
            recs = fut.result()
            if recs:
                by_task[recs[0]["task"]].extend(recs)
            if i % 10 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} batches done")

    # Per-task shuffle + split so eval covers every task type.
    train, eval_ = [], []
    for name, recs in by_task.items():
        rng.shuffle(recs)
        k = max(1, round(len(recs) * c.EVAL_FRACTION)) if recs else 0
        eval_.extend(recs[:k])
        train.extend(recs[k:])
        print(f"  {name:16s} {len(recs):4d} generated -> {len(recs) - k} train / {k} eval")

    rng.shuffle(train)
    rng.shuffle(eval_)
    write_jsonl(c.TRAIN_FILE, train)
    write_jsonl(c.EVAL_FILE, eval_)
    print(f"\nWrote {len(train)} -> {c.TRAIN_FILE}\n      {len(eval_)} -> {c.EVAL_FILE}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate HSK-5 tutoring data with Claude.")
    ap.add_argument("--dry-run", action="store_true", help="assemble + print prompts, no API calls")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task (cheap smoke test)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    args = ap.parse_args()
    run(limit=args.limit, workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
