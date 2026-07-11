"""Generate synthetic HSK-5 tutoring data with Claude (the teacher model).

For each task in config.TASKS we sample seed word(s)/grammar point(s), ask the
teacher for a batch of examples, parse them into the chat schema, then split
per-task into train/eval jsonl.

Usage:
    python gen_data.py --dry-run        # assemble + print prompts, no API calls (free)
    python gen_data.py --limit 10       # tiny paid smoke test (10 examples/task)
    python gen_data.py                  # full run (needs ANTHROPIC_API_KEY)
    python gen_data.py --only conversation   # ONE task, APPENDED to existing jsonl

Each single-turn record:
    {
      "task": "correct_sentence",
      "hsk_target": ["承担"],
      "messages": [
        {"role": "system",    "content": <SYSTEM_PROMPT>},
        {"role": "user",      "content": <learner query>},
        {"role": "assistant", "content": <ideal tutor reply>}
      ]
    }

The "conversation" task is multi-turn: messages holds a system turn
(config.conversation_system(targets) — identical to what app.py serves) followed
by 8–12 alternating user/assistant turns. One teacher call = one conversation.
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


def build_conversation_prompt(topic: str, words: list[str]) -> str:
    """Teacher prompt for ONE multi-turn conversation-practice example."""
    return f"""You are creating synthetic training data for a Chinese tutoring model \
aimed at HSK-5 learners — this task is MULTI-TURN CONVERSATION practice.

The tutor (the "assistant") MUST follow this system prompt exactly:
---
{c.conversation_system(words)}
---

Write ONE complete practice conversation between a learner ("user") and the tutor \
("assistant") about this topic: {topic}

Requirements:
- 8–12 messages total, strictly alternating roles, starting with "user" and ending \
with "assistant".
- The learner is a realistic HSK-5 student writing in Chinese. Make the learner give \
a SHORT, low-effort answer (like "还行吧。" or "没什么特别的。") at least twice, and \
make 1–2 natural learner mistakes (word order, 了/过, collocation, measure word, \
词语搭配) somewhere in the middle turns.
- The tutor must demonstrate its persona in every turn: lead the topic with its own \
opinions/experiences, push past lazy answers with concrete follow-up questions, \
correct each learner mistake briefly (correct form + ONE short English sentence why) \
and then move on, use or explicitly invite a target word in EVERY tutor turn, and end \
every tutor turn with one open-ended question. 2–4 sentences per tutor turn, HSK-5 \
Chinese only, no pinyin, and no English outside correction explanations.
- Keep the whole conversation under about 800 Chinese characters.

Return ONLY a JSON object: {{"messages": [{{"role": "user", "content": "..."}}, \
{{"role": "assistant", "content": "..."}}, ...]}}. No markdown, no commentary."""


# --------------------------------------------------------------------------- #
# Teacher call + parsing
# --------------------------------------------------------------------------- #
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def extract_json_object(text: str) -> dict:
    """Pull the JSON object out of a teacher response, tolerating stray prose or
    ```json fences."""
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def to_conversation_record(words: list[str], msgs: list[dict]) -> dict:
    """Validate + wrap one teacher conversation into the chat schema.
    Raises on malformed output so generate_conversation retries."""
    msgs = [
        {"role": m["role"], "content": m["content"].strip()}
        for m in msgs
        if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
    ]
    if len(msgs) < 6:
        raise ValueError(f"conversation too short ({len(msgs)} turns)")
    if msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        raise ValueError("conversation must start with user and end with assistant")
    if any(a["role"] == b["role"] for a, b in zip(msgs, msgs[1:])):
        raise ValueError("roles must alternate")
    total = sum(len(m["content"]) for m in msgs)
    if total > c.CONV_MAX_CHARS:
        raise ValueError(f"conversation too long ({total} chars > {c.CONV_MAX_CHARS})")
    return {
        "task": "conversation",
        "hsk_target": words,
        "messages": [{"role": "system", "content": c.conversation_system(words)}] + msgs,
    }


def generate_conversation(client, topic: str, words: list[str], retries: int = 2) -> list[dict]:
    """One API call → one multi-turn record (as a 1-element list, so results
    plug into the same per-task collection as the single-turn batches)."""
    prompt = build_conversation_prompt(topic, words)
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.messages.create(
                model=c.TEACHER_MODEL,
                max_tokens=c.GEN_MAX_TOKENS,
                temperature=c.GEN_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            data = extract_json_object(response_text(resp))
            return [to_conversation_record(words, data["messages"])]
        except Exception as e:  # noqa: BLE001 — same policy as generate_batch
            last_err = e
    print(f"  ! dropped a conversation ({topic}) after {retries + 1} tries: {last_err}")
    return []


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


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(limit: int | None, workers: int, dry_run: bool, only: str | None) -> None:
    rng = random.Random(c.TRAIN.seed)
    vocab = load_seed(c.VOCAB_FILE)
    grammar = load_seed(c.GRAMMAR_FILE)

    # Plan every teacher call up front (cheap, no API). Single-turn tasks are
    # batched EXAMPLES_PER_CALL per call; conversations are one per call.
    plan: list[tuple[c.TaskSpec, list[str]]] = []
    for task in c.TASKS:
        if only and task.name != only:
            continue
        n = min(task.n, limit) if limit else task.n
        seed = grammar if task.needs_grammar else vocab
        for batch in make_item_batches(seed, n, c.EXAMPLES_PER_CALL, rng):
            plan.append((task, batch))

    conv_plan: list[tuple[str, list[str]]] = []
    if only in (None, "conversation"):
        n = min(c.CONV_N, limit) if limit else c.CONV_N
        for words in make_item_batches(vocab, n * c.CONV_WORDS_PER, c.CONV_WORDS_PER, rng):
            conv_plan.append((rng.choice(c.CONV_TOPICS), words))

    if dry_run:
        print(f"DRY RUN — {len(plan)} single-turn batches + {len(conv_plan)} conversations planned. "
              f"Showing the first prompt per task:\n")
        seen = set()
        for task, batch in plan:
            if task.name in seen:
                continue
            seen.add(task.name)
            print(f"{'=' * 70}\nTASK: {task.name}  (targets: {batch})\n{'=' * 70}")
            print(build_teacher_prompt(task, batch))
            print()
        if conv_plan:
            topic, words = conv_plan[0]
            print(f"{'=' * 70}\nTASK: conversation  (topic: {topic}, targets: {words})\n{'=' * 70}")
            print(build_conversation_prompt(topic, words))
            print()
        return

    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    total_calls = len(plan) + len(conv_plan)
    print(f"Generating {total_calls} batches with {workers} workers "
          f"(teacher={c.TEACHER_MODEL})...")

    # Fan out the calls; group results back per task for the train/eval split.
    by_task: dict[str, list[dict]] = {t.name: [] for t in c.TASKS}
    by_task["conversation"] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_batch, client, task, batch) for task, batch in plan]
        futures += [pool.submit(generate_conversation, client, topic, words) for topic, words in conv_plan]
        for i, fut in enumerate(futures, 1):
            recs = fut.result()
            if recs:
                by_task[recs[0]["task"]].extend(recs)
            if i % 10 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} batches done")

    # Per-task shuffle + split so eval covers every task type.
    train, eval_ = [], []
    for name, recs in by_task.items():
        if not recs:
            continue
        rng.shuffle(recs)
        k = max(1, round(len(recs) * c.EVAL_FRACTION))
        eval_.extend(recs[:k])
        train.extend(recs[k:])
        print(f"  {name:16s} {len(recs):4d} generated -> {len(recs) - k} train / {k} eval")

    # --only runs APPEND to the existing data (a full run regenerates everything).
    if only:
        train = read_jsonl(c.TRAIN_FILE) + train
        eval_ = read_jsonl(c.EVAL_FILE) + eval_
        print(f"  (--only {only}: appended to existing data)")

    rng.shuffle(train)
    rng.shuffle(eval_)
    write_jsonl(c.TRAIN_FILE, train)
    write_jsonl(c.EVAL_FILE, eval_)
    print(f"\nWrote {len(train)} -> {c.TRAIN_FILE}\n      {len(eval_)} -> {c.EVAL_FILE}")


def main() -> None:
    task_names = [t.name for t in c.TASKS] + ["conversation"]
    ap = argparse.ArgumentParser(description="Generate HSK-5 tutoring data with Claude.")
    ap.add_argument("--dry-run", action="store_true", help="assemble + print prompts, no API calls")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task (cheap smoke test)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    ap.add_argument("--only", choices=task_names, default=None,
                    help="generate just this task and APPEND to the existing jsonl")
    args = ap.parse_args()
    run(limit=args.limit, workers=args.workers, dry_run=args.dry_run, only=args.only)


if __name__ == "__main__":
    main()
