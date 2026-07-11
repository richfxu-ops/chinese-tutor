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


def assign_modes(n_batches: int, rng: random.Random) -> list[str]:
    """One correction MODE per batch, distributed per config.CORRECTION_MODES.
    Short counts are padded with 'error' (keep detection sharp), then shuffled and
    trimmed to exactly n_batches so no single mode is systematically dropped."""
    modes: list[str] = []
    for mode, frac in c.CORRECTION_MODES.items():
        modes += [mode] * round(n_batches * frac)
    while len(modes) < n_batches:
        modes.append("error")
    rng.shuffle(modes)
    return modes[:n_batches]


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


# Per-mode teacher instructions for the correction task. The whole point of the
# mix is to teach the model that NOT every checked sentence has an error — the
# tuned 7B over-corrected because 100% of its correction examples fixed an error.
# See config.CORRECTION_MODES and DECISIONS 2026-07-11.
_CORRECTION_MODE_INSTRUCTIONS = {
    "error": (
        "MODE — real error: each learner sentence contains ONE realistic HSK-5 error "
        "involving the target word (word order, measure word, 了/过 aspect, a wrong "
        "collocation, a misused near-synonym, etc.). The assistant points out the "
        "problem, gives the corrected sentence, and explains the rule in one short line.\n"
        "STRICT:\n"
        "- The stated rule MUST be factually true AND consistent with your corrected "
        "sentence. Never invent a rule; never mislabel a word's part of speech or "
        "transitivity unless it truly is so.\n"
        "- The corrected sentence must genuinely fix the error and be natural, native "
        "Chinese.\n"
        "- Correct ONLY that error — do not nitpick anything else."
    ),
    "correct": (
        "MODE — already correct: each learner sentence is ALREADY fully correct, "
        "natural, native HSK-5 Chinese — NO error of any kind — and the learner asks "
        "whether it is right. The assistant CONFIRMS it is correct and briefly notes "
        "what the learner did well (the right collocation / structure / word choice).\n"
        "STRICT:\n"
        "- Do NOT invent an error and do NOT offer a 'better' version. 'Yes, this is "
        "correct' is the entire point — the model must learn to leave correct Chinese "
        "alone.\n"
        "- One short encouraging or related-usage note is allowed, but it must clearly "
        "be EXTRA info, never framed as a correction."
    ),
    "polish": (
        "MODE — awkward but not wrong: each learner sentence is GRAMMATICALLY CORRECT "
        "and fully understandable but phrased a little awkwardly / less idiomatically "
        "than a native speaker would — this is NOT an error. The assistant FIRST makes "
        "clear the sentence is correct and understandable, THEN offers a more natural "
        "phrasing as OPTIONAL polish.\n"
        "STRICT:\n"
        "- Be explicit that it is not wrong (e.g. “你这样说没有错，别人完全能听懂”), and "
        "introduce the alternative as “a more natural way to say it” — NEVER as a mistake.\n"
        "- Both the learner's version and the alternative must be correct Chinese; the "
        "alternative must be genuinely more idiomatic."
    ),
    "ambiguous": (
        "MODE — depends on intent: each learner sentence's correctness DEPENDS ON WHAT "
        "THE LEARNER MEANT — it has two valid readings, or is fine in one context but "
        "not another — and the learner asks if it is right. The assistant does NOT "
        "simply call it wrong: it says the sentence works if the learner means X, notes "
        "the other reading, and asks a brief clarifying question about the intended "
        "meaning.\n"
        "STRICT:\n"
        "- Do not fabricate an error; the point is to ask for clarification instead of "
        "over-correcting.\n"
        "- Both interpretations must be correct, natural Chinese."
    ),
}


def build_correction_prompt(mode: str, items: list[str]) -> str:
    """Teacher prompt for the correct_sentence task in one response MODE (see
    config.CORRECTION_MODES). Same JSON-array shape as build_teacher_prompt so it
    plugs into the same parse/wrap path — only the instruction body differs."""
    targets = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items))
    return f"""You are creating synthetic training data for a Chinese tutoring model \
aimed at HSK-5 learners. This batch is the "check my sentence" task.

The tutor (the "assistant") MUST follow this persona exactly:
---
{c.SYSTEM_PROMPT}
---

{_CORRECTION_MODE_INSTRUCTIONS[mode]}

Produce exactly {len(items)} examples, ONE for each target word below, in order:
{targets}

For each example:
- "user": a realistic learner message that presents a sentence using the target word \
and asks the tutor to check it (Chinese, English, or a mix — vary the phrasing).
- "assistant": the ideal tutor reply for THIS mode, following the persona and the mode \
rules above, kept at HSK-5 level. Bilingual (Chinese + English), no inline pinyin.

Return ONLY a JSON array of {len(items)} objects, each {{"user": "...", "assistant": "..."}}. \
No markdown, no commentary."""


def build_conversation_prompt(topic: str, words: list[str], has_errors: bool = True) -> str:
    """Teacher prompt for ONE multi-turn conversation-practice example. When
    has_errors is False the learner writes only correct Chinese and the tutor
    corrects nothing — teaching it to leave correct turns alone (the 聊天 half of
    the over-correction fix)."""
    if has_errors:
        learner_rule = (
            "- The learner is a realistic HSK-5 student writing in Chinese. Make the "
            "learner give a SHORT, low-effort answer (like “还行吧。” or “没什么特别的。”) "
            "at least twice, and make 1–2 natural learner mistakes (word order, 了/过, "
            "collocation, measure word, 词语搭配) somewhere in the middle turns."
        )
    else:
        learner_rule = (
            "- The learner is a realistic HSK-5 student writing in Chinese, and writes "
            "ONLY correct Chinese — plant NO mistakes anywhere. Still give a SHORT, "
            "low-effort answer (like “还行吧。” or “没什么特别的。”) at least twice so the "
            "tutor practices pushing past them. The tutor must NOT correct anything "
            "(there is nothing to correct) — it just keeps the conversation going."
        )
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
{learner_rule}
- The tutor must demonstrate its persona in every turn: lead the topic with its own \
opinions/experiences, push past lazy answers with concrete follow-up questions, and \
end every tutor turn with one open-ended question. 2–4 sentences per tutor turn, HSK-5 \
Chinese only, no pinyin, and no English outside correction explanations.
- Target words: in EVERY tutor turn, work in a target word — but PREFER inviting the \
learner to use it (e.g. “你能用‘把握’说说吗？”) or using it only where it truly fits. \
If a target word would be forced or unnatural in the tutor's own sentence, INVITE its \
use instead of cramming it in. A forced, unnatural insertion is worse than an invitation.
- Corrections are CLOSED-WORLD: the tutor corrects ONLY the mistakes you planted (if \
any), and NOTHING else — never "correct" a sentence that is already correct Chinese, \
and never invent a rule. Every correction gives the corrected sentence AND one short \
English sentence explaining the actual rule, then moves on with the topic.
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
    ```json fences — and the wrong-closing-bracket typo (…"]} → …"]) that
    otherwise throws away an entire good array."""
    match = _JSON_ARRAY_RE.search(text)
    if match:
        return json.loads(match.group(0))
    start = text.find("[")
    if start != -1:
        salvaged = re.sub(r"[}\s]+$", "]", text[start:].rstrip(), count=1)
        try:
            return json.loads(salvaged)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON array found in response: {text[:200]!r}")


def to_records(task: c.TaskSpec, items: list[str], pairs: list[dict],
               mode: str | None = None) -> list[dict]:
    """Wrap teacher {user, assistant} pairs into the final chat schema. `mode` (set
    only for correct_sentence) is recorded so the split/eval can see the response-type
    breakdown; the model never sees it."""
    if len(pairs) != len(items):
        print(f"  ~ {task.name}: teacher returned {len(pairs)} pairs for {len(items)} items (keeping {min(len(pairs), len(items))})")
    records = []
    for item, pair in zip(items, pairs):
        rec = {
            "task": task.name,
            "hsk_target": [item],
            "messages": [
                {"role": "system", "content": c.SYSTEM_PROMPT},
                {"role": "user", "content": pair["user"].strip()},
                {"role": "assistant", "content": pair["assistant"].strip()},
            ],
        }
        if mode:
            rec["correction_mode"] = mode
        records.append(rec)
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


def generate_conversation(client, topic: str, words: list[str], has_errors: bool = True,
                          retries: int = 2) -> list[dict]:
    """One API call → one multi-turn record (as a 1-element list, so results
    plug into the same per-task collection as the single-turn batches)."""
    prompt = build_conversation_prompt(topic, words, has_errors)
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


def generate_batch(client, task: c.TaskSpec, items: list[str], mode: str | None = None,
                   retries: int = 2) -> list[dict]:
    """One API call → up to len(items) records. Retries on parse failure. `mode`
    is set only for correct_sentence and selects the response-type prompt."""
    prompt = build_correction_prompt(mode, items) if mode else build_teacher_prompt(task, items)
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
            return to_records(task, items, pairs, mode)
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
    # batched EXAMPLES_PER_CALL per call; conversations are one per call. The third
    # tuple slot is the correction MODE (None for every non-correction task).
    plan: list[tuple[c.TaskSpec, list[str], str | None]] = []
    for task in c.TASKS:
        if only and task.name != only:
            continue
        n = min(task.n, limit) if limit else task.n
        seed = grammar if task.needs_grammar else vocab
        batches = make_item_batches(seed, n, c.EXAMPLES_PER_CALL, rng)
        if task.name == "correct_sentence":
            for batch, mode in zip(batches, assign_modes(len(batches), rng)):
                plan.append((task, batch, mode))
        else:
            for batch in batches:
                plan.append((task, batch, None))

    conv_plan: list[tuple[str, list[str], bool]] = []
    if only in (None, "conversation"):
        n = min(c.CONV_N, limit) if limit else c.CONV_N
        conv_batches = make_item_batches(vocab, n * c.CONV_WORDS_PER, c.CONV_WORDS_PER, rng)
        # a fixed fraction of conversations are error-free (tutor corrects nothing)
        n_free = round(len(conv_batches) * c.CONV_ERROR_FREE_FRAC)
        flags = [False] * n_free + [True] * (len(conv_batches) - n_free)
        rng.shuffle(flags)
        for words, has_errors in zip(conv_batches, flags):
            conv_plan.append((rng.choice(c.CONV_TOPICS), words, has_errors))

    if dry_run:
        n_corr = sum(1 for _, _, m in plan if m)
        print(f"DRY RUN — {len(plan)} single-turn batches ({n_corr} correction) + "
              f"{len(conv_plan)} conversations planned. Showing one prompt per task "
              f"(and per correction mode / conversation variant):\n")
        seen = set()
        for task, batch, mode in plan:
            key = (task.name, mode)
            if key in seen:
                continue
            seen.add(key)
            label = task.name + (f"  [mode={mode}]" if mode else "")
            print(f"{'=' * 70}\nTASK: {label}  (targets: {batch})\n{'=' * 70}")
            print(build_correction_prompt(mode, batch) if mode else build_teacher_prompt(task, batch))
            print()
        for has_errors in (True, False):
            match = next((cp for cp in conv_plan if cp[2] == has_errors), None)
            if match:
                topic, words, _ = match
                print(f"{'=' * 70}\nTASK: conversation  [errors={has_errors}]  "
                      f"(topic: {topic}, targets: {words})\n{'=' * 70}")
                print(build_conversation_prompt(topic, words, has_errors))
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
        futures = [pool.submit(generate_batch, client, task, batch, mode) for task, batch, mode in plan]
        futures += [pool.submit(generate_conversation, client, topic, words, has_errors)
                    for topic, words, has_errors in conv_plan]
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
        if name == "correct_sentence":
            from collections import Counter
            breakdown = Counter(r.get("correction_mode", "?") for r in recs)
            print(f"  {'':16s}      modes: {dict(breakdown)}")

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
