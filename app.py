"""Gradio chat demo for the HSK-5 tutor.

Serves the fine-tuned model (merged + quantized to GGUF — config.GGUF_FILE)
locally via llama.cpp (Metal on Apple Silicon) and
renders every reply through the reading layer (annotate.py): pinyin over each
character + hover gloss on each word.

Design note: we keep TWO copies of the conversation. The model always sees the
RAW text (feeding it our HTML back would poison the context); the chat panel
shows the ANNOTATED HTML. That separation is why the state plumbing below exists.

    python get_cedict.py      # once, for hover glosses
    python app.py             # needs outputs/hsk5-tutor-q4_k_m.gguf (see README)
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import random
import re
import threading

import gradio as gr
from llama_cpp import Llama

import config as c
from annotate import HAS_CJK, ambiguous_words, annotate, sense_options, unglossed_words
from gen_data import extract_json_array, extract_json_object, load_seed

if not c.GGUF_FILE.exists():
    raise SystemExit(
        f"Model not found: {c.GGUF_FILE}\n"
        "Build it first: train (Colab) → merge.py → GGUF quantize → put the .gguf in outputs/. "
        "See the README."
    )

# n_gpu_layers=-1 offloads everything to Metal on Apple Silicon. The chat template
# is read from the GGUF metadata (Qwen2.5 ships a ChatML-style template).
llm = Llama(model_path=str(c.GGUF_FILE), n_ctx=4096, n_gpu_layers=-1, verbose=False)

# llama-cpp is NOT thread-safe, and Gradio 6 runs each event listener on its own
# concurrency queue — a card-example request can fire while respond() streams.
# Every model call goes through this lock; the streaming loop holds it for the
# whole generation, so other events simply wait their turn.
_LLM_LOCK = threading.Lock()


def generate(messages: list[dict], **kw) -> str:
    """Serialized non-streaming completion — the one way to call the model."""
    with _LLM_LOCK:
        out = llm.create_chat_completion(messages, **kw)
    return out["choices"][0]["message"]["content"]

# Fallback starter pool, ~4 per task type — used only when gen_starters()
# below fails. Normally the starters are written by the model at startup,
# seeded with random vocab/grammar so every launch gets a genuinely new set.
STARTER_POOL = [
    # explain_word
    "“毕竟”是什么意思？",
    "解释一下“临时”这个词",
    "“居然”和“竟然”有什么区别？",
    "“把握”是什么意思？怎么用？",
    # use_in_sentence
    "用“承担”造两个句子",
    "用“逐渐”造个句子",
    "用“珍惜”写三个例句",
    "用“体会”造句",
    # correct_sentence
    "帮我改这个句子：我昨天去过北京。",
    "这个句子对吗？我很喜欢吃中国菜非常。",
    "帮我改一下：他比我更高很多。",
    "这样说对吗？我明天要见面我的朋友。",
    # translate
    "How do I say “I can’t help but worry” in Chinese?",
    "怎么用中文说 “it’s not worth it”？",
    "Translate: 这件事说起来容易，做起来难。",
    "How do I say “long time no see” in a formal way?",
    # example_dialogue
    "给我一段在餐厅点菜的对话",
    "给我一段面试时的对话",
    "写一段在机场值机的对话",
    "给我一段去医院看病的对话",
    # grammar_point
    "怎么用“既然……就……”这个语法？",
    "“除非……否则……”怎么用？",
    "“越来越”和“越……越……”有什么不同？",
    "“连……都……”是什么意思？",
]


def _load_seed(path) -> list[str]:
    return load_seed(path) if path.exists() else []


# Per-level seed lists (starters, conversation targets, reading passages).
# A missing file just means an empty pool — every consumer degrades gracefully.
LEVEL_VOCAB = {lvl: _load_seed(p) for lvl, p in c.LEVEL_VOCAB_FILES.items()}
LEVEL_GRAMMAR = {lvl: _load_seed(p) for lvl, p in c.LEVEL_GRAMMAR_FILES.items()}


def _level_num(label: str | int) -> int:
    """The HSK level from the UI radio's label ('HSK 4' → 4); anything odd → 5."""
    m = re.search(r"[456]", str(label or ""))
    return int(m.group()) if m else 5


def gen_starters(level: int = 5) -> list[str]:
    """Model-written starter chips, one per task type, seeded with random vocab
    and grammar so every launch is a new set. Falls back to STARTER_POOL —
    for ANY failure, including missing/short seed files, so this can never
    block launch (everything lives inside the try)."""
    try:
        words = random.sample(LEVEL_VOCAB[level], 4)
        grammar = random.choice(LEVEL_GRAMMAR[level])
        prompt = (
            f"为一个HSK{level}水平的学生写6个开场问题，模拟学生问中文老师时会说的话，每种一个。"
            "每一条都必须是学生对老师的【请求或提问】，不能是回答或对话本身：\n"
            f"1. 问“{words[0]}”是什么意思\n"
            f"2. 请老师用“{words[1]}”造句\n"
            f"3. 格式：“这个句子对吗？……”——句子由你编，含一个典型的学习者语法错误，用上“{words[2]}”\n"
            "4. 问一个日常英文表达用中文怎么说，格式：“How do I say ‘…’ in Chinese?”"
        "——引号里必须填英文表达（选一个不能直译的，绝对不能填中文）\n"
            f"5. 格式：“给我一段关于…的对话”，场景跟“{words[3]}”有关\n"
            f"6. 问“{grammar}”这个语法怎么用\n"
            "要求：口语化、简短（每个不超过25个字）。"
            '只返回一个JSON数组，包含6个字符串，例如 ["…","…","…","…","…","…"]。不要解释。'
        )
        # 600, not 400: six starters + JSON syntax can crest 400 tokens, and a
        # truncated array (no closing ]) fails extraction → needless fallback
        out = generate([{"role": "user", "content": prompt}],
                       temperature=0.8, max_tokens=600)
        arr = extract_json_array(out)
        starters = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
        # a "How do I say X in Chinese?" starter must not contain Chinese —
        # the model occasionally fills the slot with a Chinese phrase, which
        # is nonsense by construction
        starters = [s for s in starters
                    if not ("how do i say" in s.lower() and HAS_CJK.search(s))][:6]
        if len(starters) >= 4:
            if len(starters) < 6:   # top up from the pool so the row stays full
                starters += random.sample(
                    [p for p in STARTER_POOL if p not in starters], 6 - len(starters))
            return starters
        raise ValueError(f"only {len(starters)} usable starters in: {out[:120]!r}")
    except Exception as e:  # noqa: BLE001 — starters are decoration, never block launch
        print(f"starters: generation failed ({e}); using the fallback pool", flush=True)
    return random.sample(STARTER_POOL, 6)


print("writing fresh starter prompts...", flush=True)
STARTERS = gen_starters()


# The transcript is rendered as raw HTML via gr.HTML — NOT gr.Chatbot. Gradio 6's
# Chatbot markdown pass sanitizes messages and strips the `title` attribute (and
# non-standard tags), which kills the hover gloss. gr.HTML renders our reading-layer
# markup untouched (verified: <style>, ruby/rt and attributes all survive).
#
# ---- Look: "teacher's red ink" (朱批) — a scholar's paper study ----
# Warm rice-paper page with a grain texture, ink typography (EB Garamond for
# Latin, Songti/Kaiti for hanzi), and ONE accent: cinnabar red — the color a
# Chinese teacher marks corrections in. Pinyin, the seal, the user's wash and
# every hover all draw from it. Global CSS goes to launch(css=...) (Gradio 6
# moved theme/css/js there); the theme object handles fonts + base variables.
#
# Two Gradio-specific quirks this CSS works around:
#  - Gradio ships `.gradio-container-* .prose * { color: var(--body-text-color) }`,
#    a universal selector that recolors every element and beats inherited colors —
#    the `!important` on the .msg color rules is what keeps the transcript readable.
#  - Native `title` tooltips are slow (~1s delay) and unreliable inside embedded
#    webviews, so the gloss is shown with a styled CSS tooltip instead: render_chat
#    rewrites annotate()'s title= to data-tip= and .hz:hover::after displays it.


# The page CSS lives in web/app.css (plain CSS — see its header note).
PAGE_CSS = (c.ROOT / "web" / "app.css").read_text(encoding="utf-8")

# Page JS, run once at load. Injected via launch(head=...) as a self-invoking
# <script> — Gradio 6.20 ships launch(js=...) into the frontend config but never
# invokes it (verified empirically), so head= is the reliable hook.
#  - Light-only: the app is designed as paper; Gradio follows the OS and may add
#    .dark — strip it and keep it off (the :root,.dark CSS vars are the backstop).
#  - Click-to-collect: clicking any annotated word saves {word, pinyin, gloss,
#    example sentence} into the flashcards deck (localStorage, same key/schema as
#    web/flashcards.html, so the review tab reads the same deck). The transcript
#    HTML is replaced every turn, so the listener is delegated from document.
# The page JS lives in web/app.js (it keeps the __STARTERS__ placeholder
# that HEAD_HTML below fills in).
APP_JS = (c.ROOT / "web" / "app.js").read_text(encoding="utf-8")

HEAD_HTML = f"<script>{APP_JS.replace('__STARTERS__', json.dumps(STARTERS, ensure_ascii=False))}</script>"

HEADER_HTML = (c.ROOT / "web" / "header.html").read_text(encoding="utf-8")

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fbeae8", c100="#f6d5d1", c200="#eaaba3", c300="#dd8175",
        c400="#cc5a4c", c500="#b3302a", c600="#9c2a25", c700="#8f241f",
        c800="#731d19", c900="#571613", c950="#3a0f0d",
    ),
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("EB Garamond"), "Songti SC", "serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)


# For the per-bubble TTS button: keep only the Chinese of a bilingual reply
# (per line-with-CJK, drop the Latin words) so the zh voice doesn't wade
# through the English translations.
# A run of hanzi plus any digits/CJK punctuation glued to it — punctuation kept
# so the voice pauses naturally, digits so it reads numbers inside a sentence.
_CJK_CHUNK = re.compile(r"[一-鿿]+[0-9，。！？、；：]*")


def chinese_only(text: str) -> str:
    kept = []
    for line in text.split("\n"):
        if HAS_CJK.search(line):
            kept.append("".join(_CJK_CHUNK.findall(line)))
    return " ".join(kept)


# List markers / labels the model sometimes prefixes to its lines ("1. ", "例句：").
# Stripped by the translators (translate_user_to_chinese/_english) and by
# gen_card_example when parsing the model's line-formatted replies.
_UNLABEL = re.compile(
    r"^\s*(?:\d+[.、)]|[-•*]|第[一二三]行[:：]?|例句[:：]?|释义[:：]?|翻译[:：]?|Translation[:：]?)\s*")
# A trailing run of English glued onto a Chinese line (我很高兴，I am happy).
# Two users: enforce_chinese_reply strips it as 聊天-mode translation drift, and
# gen_card_example trims it off example sentences so the zh TTS never voices it
# (the translation belongs in example_en).
_TRAILING_EN = re.compile(r"[，,\s]*[A-Za-z][A-Za-z0-9 ,.'’!?;:-]{7,}[.!?]?\s*$")


# A tutor correction: 应该说“正确的句子”，因为解释… (the trained shape) or
# 解释…，所以要说“正确的句子”。 The corrected sentence is the quoted group; the
# explanation is the rest of the sentence around the match (either side), plus
# the following sentence when it's the (mostly-ASCII) English rule. The
# (?<![不别]) guard skips 不要说/别说“错的” so we never capture the wrong
# version as the fix.
_FIX_RE = re.compile(r"(?<![不别])(?:应该|要)\s*说\s*[“\"]([^”\"]+)[”\"]")
_SENT_END = re.compile(r"[。！？!?.]")
# Praise/agreement in the same sentence means the quote is NOT a correction
# (你说得对，应该说“…”也可以) — no chip.
_NOT_A_FIX = re.compile(r"说得对|说得很好|没问题|没有错|是对的|很自然|用得很好")
# A real correction REVISES the student's sentence, so ≥60% of the fix's unique
# hanzi must come from it. Validated against the training set (144/150 trained
# corrections pass — DECISIONS.md 2026-07-11); re-measure before changing.
_FIX_OVERLAP_MIN = 0.6


# ">60% of non-space chars are ASCII" — the working boundary between "an English
# line" and "a Chinese line that merely contains some English".
_MOSTLY_ASCII_MIN = 0.6


def _mostly_ascii(s: str) -> bool:
    letters = [ch for ch in s if not ch.isspace()]
    return bool(letters) and sum(ch.isascii() for ch in letters) / len(letters) > _MOSTLY_ASCII_MIN


def extract_correction(reply: str, wrong: str) -> tuple[str, str] | None:
    """(corrected sentence, explanation) if the reply corrects `wrong`, the
    student's previous message.

    Guards against false chips on benign 应该说/要说 quoting: a real correction
    REVISES the student's sentence, so the quoted fix must be built mostly from
    the student's own characters, and the sentence must not be praise."""
    m = _FIX_RE.search(reply)
    if not m:
        return None
    ends = [e.end() for e in _SENT_END.finditer(reply, 0, m.start())]
    sent_start = ends[-1] if ends else 0
    end_m = _SENT_END.search(reply, m.end())
    sent_end = end_m.start() if end_m else len(reply)
    if _NOT_A_FIX.search(reply[sent_start:sent_end]):
        return None
    fix_chars = {ch for ch in m.group(1) if HAS_CJK.match(ch)}
    if len(fix_chars) < 2 or not wrong:
        return None
    if len(fix_chars & set(wrong)) / len(fix_chars) < _FIX_OVERLAP_MIN:
        return None
    pre = reply[sent_start:m.start()].strip().strip("，、： ")
    if len(pre) <= 3:   # a bare subject ("你"/"这里"), not an explanation
        pre = ""
    post = reply[m.end():sent_end].strip().strip("，、： ")
    why = "，".join(p for p in (pre, post) if p)
    for tail in ("，所以要", "，所以", "所以要", "所以"):   # leftovers of "…，所以要说“X”"
        if why.endswith(tail):
            why = why[: -len(tail)]
            break
    if end_m:  # append the English rule when the next sentence is mostly ASCII
        rest = reply[end_m.end():]
        nxt_end = _SENT_END.search(rest)
        nxt = (rest[: nxt_end.start()] if nxt_end else rest).strip()
        if _mostly_ascii(nxt):
            why = f"{why}。{nxt}" if why else nxt
    return m.group(1), why[:200]


# --------------------------------------------------------------------------- #
# In-context sense disambiguation (地道: tunnel or authentic?). One extra short
# generation per turn, only when the new messages contain ambiguous words. The
# result feeds annotate() overrides, so tooltip AND ruby pinyin reflect the
# sense actually used. Failures fall back silently to the dictionary tips.
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]")


def _tipped(text: str, ov: dict[str, tuple[list[str], str]] | None = None) -> str:
    """annotate() for in-app rendering: the native `title` tooltip (right for
    the standalone docs/ pages) becomes our styled CSS tooltip's data-tip.
    Safe: message text is html-escaped, so ` title="` can only come from
    annotate()'s own .hz spans."""
    return annotate(text, ov).replace(' title="', ' data-tip="')


def _sentence_around(text: str, word: str) -> str:
    for chunk in _SENTENCE_SPLIT.split(text):
        if word in chunk:
            return chunk.strip()[:80]
    return text.strip()[:80]


def disambiguate(texts: list[str]) -> dict[str, tuple[list[str], str]]:
    """One model call → tooltip overrides {word: (syllables, gloss)} for the
    new messages: sense PICKS for ambiguous words (地道: option number) and
    written DEFINITIONS for words the dictionary lacks entirely (一只). One
    flat JSON does both — a number means a pick, a string means a definition."""
    sense: dict[str, tuple[str, list[dict]]] = {}   # word -> (sentence, options)
    defs: dict[str, str] = {}                       # word -> sentence
    for text in texts:
        for w in ambiguous_words(text):
            if w not in sense and len(sense) < 6:
                sense[w] = (_sentence_around(text, w), sense_options(w))
        for w in unglossed_words(text):
            if w not in defs and len(defs) < 4:
                defs[w] = _sentence_around(text, w)
    if not sense and not defs:
        return {}
    blocks = []
    for w, (sentence, opts) in sense.items():
        lines = "\n".join(f"{i + 1}. {o['reading']} — {o['brief']}" for i, o in enumerate(opts))
        blocks.append(f"词：{w}\n句子：{sentence}\n{lines}")
    if defs:
        blocks.append(
            "下面的词没有选项——给每个词写一个简短的英文释义（词典格式，"
            "比如 to meet; to see each other）：\n"
            + "\n".join(f"词：{w}（句子：{s}）" for w, s in defs.items())
        )
    ex_pick = f'"{next(iter(sense))}": 1' if sense else ""
    ex_def = f'"{next(iter(defs))}": "a short English definition"' if defs else ""
    example = ", ".join(x for x in (ex_pick, ex_def) if x)
    prompt = (
        "你是词典助手。有选项的词：判断它在句子里用的是哪个意思，返回选项编号。"
        "没有选项的词：直接写英文释义。\n\n"
        + "\n\n".join(blocks)
        + "\n\n只返回一个JSON对象，例如 {" + example + "}。不要解释。"
    )
    try:
        out = generate([{"role": "user", "content": prompt}],
                       temperature=0, max_tokens=220)
        picks = extract_json_object(out)
        overrides = {}
        for w, v in picks.items():
            if w in sense and isinstance(v, (int, str)) and str(v).isdigit():
                opts = sense[w][1]
                i = int(v) - 1
                if 0 <= i < len(opts):
                    overrides[w] = (opts[i]["syllables"], opts[i]["gloss"])
            elif w in defs and isinstance(v, str) and v.strip() and _mostly_ascii(v):
                # empty syllables keep the pypinyin ruby; only the gloss changes
                overrides[w] = ([], v.strip()[:80])
        return overrides
    except Exception:  # noqa: BLE001 — a failed pick just keeps dictionary tips
        return {}


# --------------------------------------------------------------------------- #
# Q&A translation fill: the tuned model SOMETIMES skips the English line under
# a Chinese sentence. Rather than retrain (or bolt on a cloud API), one extra
# local call translates just the missed lines — only on turns that have any.
# --------------------------------------------------------------------------- #
_CJK_COUNT = re.compile(r"[一-鿿]")
_INLINE_EN = re.compile(r"[A-Za-z][A-Za-z ,.'’!?;:-]{7,}")
_NUMBERED = re.compile(r"\s*(\d+)[.、)]\s*(.+)")


def untranslated_lines(reply: str) -> dict[int, str]:
    """{line index: Chinese line} for substantial Chinese lines with no inline
    English and no mostly-English line following them."""
    lines = reply.split("\n")
    out: dict[int, str] = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if len(_CJK_COUNT.findall(s)) < 4:      # empty / label / fragment
            continue
        if _INLINE_EN.search(s):                # translated inline already
            continue
        nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        if nxt and _mostly_ascii(nxt):          # translated on the next line
            continue
        out[i] = s
    return out


def fill_translations(reply: str) -> dict[int, str]:
    """One model call → {line index: English} for the reply's missed lines."""
    missing = untranslated_lines(reply)
    if not missing:
        return {}
    numbered = "\n".join(f"{i}. {s}" for i, s in missing.items())
    prompt = (
        "把下面每行中文翻译成英文。保持行号，逐行返回，格式：行号. 英文翻译。"
        "不要拼音，不要解释。\n\n" + numbered
    )
    try:
        out = generate([{"role": "user", "content": prompt}],
                       temperature=0, max_tokens=60 * len(missing) + 40)
        fills = {}
        for line in out.splitlines():
            m = _NUMBERED.match(line)
            if m and int(m.group(1)) in missing and _mostly_ascii(m.group(2)):
                fills[int(m.group(1))] = m.group(2).strip()[:200]
        return fills
    except Exception:  # noqa: BLE001 — a failed fill just leaves the reply as-is
        return {}


def translate_user_to_chinese(text: str, max_tokens: int = 120) -> str:
    """Natural HSK-5 Chinese for English text — the student's prompt (shown
    annotated under their bubble), or a fully-English reply the conversation
    tutor should never have produced (converted before display/history)."""
    prompt = (
        "把下面这段话翻译成自然的中文（HSK5水平的说法），忠实传达原意。"
        "这是纯翻译任务：原文是问题就翻译问题本身，是请求或指令"
        "（比如“写几个例句”“帮我改句子”）就翻译这个请求本身——"
        "绝对不要回答问题，也不要执行请求。"
        "比如原文问某个词或短语“用中文怎么说”，要保留这种提问的形式"
        "（引号里的英文原样保留），不要给出答案。"
        "只返回中文翻译，不要拼音，不要解释。\n\n原文：" + text.strip()
    )
    try:
        out = generate([{"role": "user", "content": prompt}],
                       temperature=0, max_tokens=max_tokens)
        lines = [_UNLABEL.sub("", l).strip().strip('"“”')
                 for l in out.splitlines() if l.strip()]
        zh = "\n".join(l for l in lines if HAS_CJK.search(l))
        # answered-instead-of-translated backstop: a real translation can't be
        # several times longer than the source ("write three sentences" → three
        # actual sentences). No translation beats a wrong one.
        if len(HAS_CJK.findall(zh)) > max(30, 3 * len(text.split())):
            return ""
        return zh[:400] if zh else ""
    except Exception:  # noqa: BLE001 — no translation is just the old behavior
        return ""


# "Sure, here's the translation:" / "In English this translates to:" — a short
# lead-in ending in translat* (+ optional is/to/as/would/be) and a colon. Needs
# the word itself, so a translation merely starting with "Sure" is never eaten;
# only linking words may sit between it and the colon, so a REAL translation
# like "Translate this document: ..." survives. No .!?/quotes in the lead-in —
# a preamble is one short clause, not a finished sentence.
_EN_PREAMBLE = re.compile(
    r"^[^\"“”.!?]{0,40}?translat\w*(?:\s+(?:is|to|as|would|be)){0,2}\s*[:：]\s*",
    re.IGNORECASE)


def translate_user_to_english(text: str, max_tokens: int = 160, max_chars: int = 400) -> str:
    """Faithful English translation of Chinese `text`, at temperature 0.
    Two users: the fill under the student's bubble (defaults sized for a chat
    message) and the reading passage's translation (larger budgets, passed in) —
    translating the FINAL text in its own call is what keeps the translation
    matching the text."""
    prompt = (
        "把下面这段中文翻译成自然的英文，忠实传达原意。"
        "这是纯翻译任务：原文是问题就翻译问题本身，是请求或指令"
        "（比如“用‘珍惜’写三个例句”要译成 Write three example sentences using 珍惜）"
        "就翻译这个请求本身——绝对不要回答问题，也不要执行请求。"
        "直接输出译文本身：不要任何开场白、说明或引号"
        "（不要写 Sure、Here's the translation 之类）。\n\n原文：" + text.strip()
    )
    try:
        out = generate([{"role": "user", "content": prompt}],
                       temperature=0, max_tokens=max_tokens)
        lines = [_UNLABEL.sub("", l).strip().strip('"“”')
                 for l in out.splitlines() if l.strip()]
        en = " ".join(l for l in lines if _mostly_ascii(l))
        # belt-and-braces: drop a leading "Sure, here's the translation:"-style
        # preamble the model sometimes adds despite the prompt (anything short
        # ending in translation/translates-to + colon)
        stripped = _EN_PREAMBLE.sub("", en).strip()
        en = (stripped or en)[:max_chars]
        # answered-instead-of-translated backstop (mirror of the zh direction):
        # an 8-hanzi request can't translate to 35 English words
        if len(en.split()) > max(12, 3 * len(HAS_CJK.findall(text))):
            return ""
        return en
    except Exception:  # noqa: BLE001 — no translation is just the old behavior
        return ""


def _spk(line: str) -> str:
    """A small per-line TTS button voicing just that line's Chinese."""
    return (f'<button class="spk line" title="朗读这一行 · read this line aloud"'
            f' data-speak="{html.escape(chinese_only(line), quote=True)}">🔊</button>')


# A run of Latin text long enough to be a translation/phrase (not a short inline
# loanword like a quoted 'apple'), used to break a bilingual line in two.
_EN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9 ,.;:!?'’\"()/%–—-]{7,}")


def _split_bilingual_line(line: str) -> list[str]:
    """Break one display line so a Chinese sentence and its trailing English
    translation don't share a line — the model sometimes glues them (or a
    trailing 'Keep it up!'). We split only before an English run that FOLLOWS a
    finished Chinese sentence (。！？), so an English phrase quoted mid-sentence
    (他说“I am happy”的意思…) and short inline loanwords stay put, and an English
    line that itself quotes a Chinese word (…take 很多 directly) is kept whole.
    Pure-Chinese / pure-English lines pass through unchanged."""
    if not (HAS_CJK.search(line) and _EN_RUN.search(line)):
        return [line]
    for mt in _EN_RUN.finditer(line):
        head = line[:mt.start()]
        h = head.rstrip(" 　\"“”'‘’")          # ignore quotes/spaces before the run
        if h and h[-1] in "。！？!?" and HAS_CJK.search(h):
            return [head.strip(" 　"), line[mt.start():].strip(" 　")]
    return [line]


def _render_msg(m: dict, tutor: bool) -> str:
    """A message's display HTML: annotated text line by line, a 🔊 per Chinese
    line (tutor lines, and Chinese fills anywhere), plus any filled-in
    translations under their lines. Fills go through the reading layer too —
    an English fill passes through untouched, a Chinese one (the translation
    of the student's English prompt) gets ruby + glosses + click-to-collect.
    m["display"], when present, is what the student actually typed — shown
    instead of m["content"] (which is what the MODEL sees, e.g. the Chinese
    translation of an English prompt in conversation mode)."""
    ov, fills = m.get("tips"), m.get("fills") or {}
    parts = []
    for i, line in enumerate(m.get("display", m["content"]).split("\n")):
        # keep Chinese and its English translation on separate lines
        for seg in _split_bilingual_line(line):
            h = _tipped(seg, ov)
            if tutor and HAS_CJK.search(seg):
                h += _spk(seg)
            parts.append(h)
        if i in fills:
            f = _tipped(fills[i], ov)
            if HAS_CJK.search(fills[i]):
                f += _spk(fills[i])
            parts.append(f'<span class="fill">{f}</span>')
    return "<br>".join(parts)


def render_chat(raw: list[dict], unclosed: bool = False) -> str:
    """Build the whole transcript as one self-contained HTML block, annotating every
    Chinese span (pinyin ruby + hover gloss). A message's disambiguation
    overrides travel ON the message dict (m["tips"]) so they can never
    misalign; strip_for_llm() removes them before generation.

    unclosed=True omits the closing wrapper tag so the streaming path can
    append its live bubble — the open/close contract lives HERE, not in
    string surgery at the call site."""
    bubbles = []
    for i, m in enumerate(raw):
        u = m["role"] == "user"
        who = "You" if u else "老师 Tutor"
        msg = _render_msg(m, tutor=not u)
        # When the tutor corrected the previous student turn, offer to save the
        # correction as a flashcard (front: the student's sentence, back: the
        # fix + rule). APP_JS handles the click; the data- attrs carry the card.
        chip = ""
        if not u and i and raw[i - 1]["role"] == "user":
            fix = extract_correction(m["content"], raw[i - 1]["content"])
            if fix:
                chip = (
                    f'<button class="fix-collect"'
                    f' data-wrong="{html.escape(raw[i - 1]["content"].strip()[:160], quote=True)}"'
                    f' data-fix="{html.escape(fix[0], quote=True)}"'
                    f' data-why="{html.escape(fix[1], quote=True)}">'
                    f'✚ 收藏纠错 · save correction</button>'
                )
        bubbles.append(
            f'<div class="b {"u" if u else "t"}"><div class="who">{who}</div>'
            f'<div class="msg">{msg}</div>{chip}</div>'
        )
    inner = "".join(bubbles) or '<div class="empty">用中文或英文问我… (ask me anything)</div>'
    out = f'<div class="chat">{inner}'
    return out if unclosed else out + "</div>"


# Target words for conversation mode: the student's own collected flashcard
# words first (the deck lives client-side, so APP_JS mirrors the word list into
# a hidden textbox), padded to 5 with random seeds from the selected level's
# vocab list. Picked once per conversation and kept until 清空 so the tutor can
# keep circling back to them.
def pick_targets(deck_json: str, review_only: bool = False, level: int = 5) -> list[str]:
    """Conversation target words. Normal mode: up to 3 deck words (due first)
    padded to 5 with random seeds at the selected HSK level. Review mode: ONLY
    due cards — the conversation becomes the SRS review session (falls back to
    normal when nothing is due)."""
    def clean(ws):
        return [w for w in ws if isinstance(w, str) and w.strip()]
    try:
        data = json.loads(deck_json or "{}")
    except (json.JSONDecodeError, TypeError):
        data = {}
    if isinstance(data, list):                       # pre-split mirror shape
        due, other = clean(data), []
    else:
        due, other = clean(data.get("due", [])), clean(data.get("other", []))
    if review_only and due:
        return due[:5]
    targets = (due + other)[:3]
    pool = [w for w in LEVEL_VOCAB.get(level, LEVEL_VOCAB[5]) if w not in targets]
    targets += random.sample(pool, k=min(5 - len(targets), len(pool)))
    return targets


def render_targets(targets: list[str] | None) -> str:
    if not targets:
        return ""
    chips = "".join(f'<span class="tg">{_tipped(w)}</span>' for w in targets)
    return f'<div class="targets-row"><span class="targets-label">目标词 · practice</span>{chips}</div>'


def strip_for_llm(raw: list[dict]) -> list[dict]:
    """History for the model: role/content only — display metadata like the
    per-message "tips" overrides never reaches the chat template."""
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def enforce_chinese_reply(reply: str) -> str:
    """聊天-mode contract: English belongs ONLY inside a correction. When a
    reply contains no correction, translation drift (a standalone English
    line, or an English run glued to the end of a Chinese line) is stripped —
    from history too, so drift doesn't self-reinforce as fake context. A reply
    WITH a correction is left untouched: its English rule explanation is
    legitimate and too entangled to edit safely.

    Returns "" when nothing Chinese survives (a fully-English reply) — the
    caller then CONVERTS the reply instead of displaying English."""
    if _FIX_RE.search(reply):
        return reply
    kept = []
    for line in reply.split("\n"):
        s = line.strip()
        if s and _mostly_ascii(s) and not HAS_CJK.search(s):
            continue                                    # standalone translation line
        if HAS_CJK.search(s):
            line = _TRAILING_EN.sub("", line.rstrip()) or line
        kept.append(line)
    return "\n".join(kept).strip()


def _fill_under_last_line(turn: dict, typed: str, translation: str) -> None:
    """Attach `translation` as a fill under the last NON-EMPTY line of what the
    bubble displays — a trailing newline must not detach it."""
    lines = typed.split("\n")
    idx = max((i for i, l in enumerate(lines) if l.strip()), default=0)
    turn["fills"] = {idx: translation}


def respond(user_msg: str, raw: list[dict], mode: str, deck_json: str,
            targets: list[str] | None, review_only: bool, level_label: str = "HSK 5"):
    """user message + raw history → model reply, STREAMED.

    A generator: tokens render live in a plain bubble (the reading layer needs
    whole words, so annotating a half-generated sentence would misparse), then
    a short 加注中 pass runs disambiguation and translation fills, and the
    final yield swaps in the fully annotated transcript.
    Yields (chat HTML, raw history, cleared box, targets state, targets bar)."""
    conversational = "聊天" in mode

    def bar() -> str:
        return render_targets(targets if conversational else None)

    if not user_msg.strip():
        yield (render_chat(raw), raw, "", targets, bar())
        return
    level = _level_num(level_label)
    if conversational:
        if not targets:
            targets = pick_targets(deck_json, review_only, level)
        # at level 5 this is the exact prompt the conversation data was trained
        # on (shared with gen_data.py); 4/6 are prompt-steered variants
        system = c.conversation_system(targets, level)
    else:
        system = c.system_prompt_app(level)
    user_turn = {"role": "user", "content": user_msg}
    # Translate when the message is at least half English by content: catches pure
    # and English-dominant input in both modes (the old *mostly-ASCII* gate skipped
    # short/mixed messages), but leaves a predominantly-Chinese message with a stray
    # English word alone — otherwise 聊天 would round-trip the student's own Chinese.
    latin_count = len(re.findall(r"[A-Za-z]", user_msg))
    msg_is_english = latin_count >= 4 and latin_count >= len(HAS_CJK.findall(user_msg))
    # the mirror: a Chinese message gets its English beneath. Same ≥4-char bar;
    # a real sentence, not a 你好.
    msg_is_chinese = not msg_is_english and len(HAS_CJK.findall(user_msg)) >= 4
    raw = raw + [user_turn]
    # the message lands in the transcript immediately, and its translation is
    # rendered BEFORE the reply generates (user preference: see the translation
    # first) — at the cost of one translation call (~1–2s) before streaming starts
    yield (render_chat(raw), raw, "", targets, bar())
    if msg_is_english:
        # English input → its Chinese beneath. In 聊天 mode the MODEL must also
        # see the Chinese instead of the English — English input makes the tutor
        # drift into translating/evaluating instead of conversing.
        zh = translate_user_to_chinese(user_msg)
        if zh:
            _fill_under_last_line(user_turn, user_msg, zh)
            if conversational:
                user_turn["content"] = zh
                user_turn["display"] = user_msg
    elif msg_is_chinese:
        # Chinese input → its English beneath ("did I say what I meant?")
        en = translate_user_to_english(user_msg)
        if en:
            _fill_under_last_line(user_turn, user_msg, en)
    if user_turn.get("fills"):
        yield (render_chat(raw), raw, "", targets, bar())
    messages = [{"role": "system", "content": system}] + strip_for_llm(raw)

    # history annotated once; the in-progress reply rides in a plain bubble
    base = render_chat(raw, unclosed=True)

    def with_stream_bubble(text: str, note: str = "") -> str:
        inner = html.escape(text).replace("\n", "<br>")
        cursor = "" if note else '<span class="cursor">▌</span>'
        who = "老师 Tutor" + (f'<span class="note"> · {note}</span>' if note else "")
        return (base + f'<div class="b t"><div class="who">{who}</div>'
                f'<div class="msg plain">{inner}{cursor}</div></div></div>')

    reply = ""
    try:
        n = 0
        with _LLM_LOCK:   # held for the whole generation — see _LLM_LOCK's note
            for chunk in llm.create_chat_completion(
                messages, temperature=0.7, max_tokens=512, stream=True,
            ):
                delta = chunk["choices"][0]["delta"].get("content")
                if not delta:
                    continue
                reply += delta
                n += 1
                if n % 8 == 0:
                    yield (with_stream_bubble(reply), raw, "", targets, bar())
    except Exception:  # noqa: BLE001 — a partial reply is kept; total failure below
        pass
    reply = reply.strip()
    if not reply:
        # honest failure: nothing enters history, the typed message goes back in
        # the box for a retry, and the error bubble is display-only (transient)
        err = with_stream_bubble(
            "生成失败了——按发送再试一次。The model errored; press send to retry.",
            note="出错 error",
        )
        yield (err, raw[:-1], user_msg, targets, bar())
        return
    if conversational:
        cleaned = enforce_chinese_reply(reply)
        if cleaned and HAS_CJK.search(cleaned):
            reply = cleaned
        else:
            # the model went fully English — convert its reply to Chinese
            # rather than let 聊天 mode speak English
            yield (with_stream_bubble(reply, note="翻译中 translating"), raw, "", targets, bar())
            reply = translate_user_to_chinese(reply, max_tokens=300) or reply
    raw = raw + [{"role": "assistant", "content": reply}]

    # reading-layer pass: sense picks + reply translation fills (the INPUT
    # translation already happened before generation). disambiguate sees what
    # the transcript shows — raw[-2]["content"] is the Chinese translation when
    # the student typed English in 聊天 mode.
    yield (with_stream_bubble(reply, note="加注中 annotating"), raw, "", targets, bar())
    ov = disambiguate([raw[-2]["content"], reply])
    if ov:
        raw[-2]["tips"] = ov
        raw[-1]["tips"] = ov
    # Bilingual replies in BOTH modes: every untranslated Chinese line in the
    # tutor's reply gets its English translation. (聊天 was Chinese-only by design;
    # now consistent with Q&A per user request. The MODEL still writes clean Chinese
    # — the English is a serve-layer fill, exactly like Q&A — so the reading layer
    # and the correction-detection stay unaffected.)
    fills = fill_translations(reply)
    if fills:
        raw[-1]["fills"] = fills
    # (input translations — English→Chinese and Chinese→English — happen BEFORE
    # generation now, so the student sees them first; nothing to add here)
    yield (render_chat(raw), raw, "", targets, bar())


def gen_card_example(req_json: str) -> str:
    """Write a fresh HSK-5 example sentence for a just-collected flashcard —
    and, when the dictionary had no gloss for the word (CEDICT lacks many
    jieba compounds like 一只/很累), a brief English definition too.

    Called through a hidden request textbox (APP_JS writes {id, word, gloss}
    on collect); the result lands in a hidden HTML div the page JS observes,
    and fills the card. Runs on the same llm/queue as chat, so it simply
    waits its turn behind a generation."""
    try:
        req = json.loads(req_json)
        card_id, word, gloss = req["id"], req["word"], req.get("gloss", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""
    if gloss:
        prompt = (
            f"用“{word}”（意思：{gloss}）写一个HSK5水平的简单例句，能自然地体现这个词的意思。"
            "第一行：例句（不要拼音）。第二行：这个例句的英文翻译。只返回这两行，不要解释。"
        )
    else:
        prompt = (
            f"“{word}”是一个中文词或词组。\n"
            f"第一行：它的简短英文释义（词典格式，比如 to meet; to see each other）。\n"
            f"第二行：用“{word}”写一个HSK5水平的简单例句（不要拼音）。\n"
            "第三行：例句的英文翻译。只返回这三行，不要解释。"
        )
    out = generate([{"role": "user", "content": prompt}],
                   temperature=0.7, max_tokens=160).strip()
    lines = [_UNLABEL.sub("", line).strip().strip('"“”')
             for line in out.splitlines() if line.strip()]
    # the example is the first mostly-CHINESE line containing the word (a
    # definition line echoing the headword, '很累 — to be very tired', is
    # mostly ASCII and must not qualify); the definition is an English line
    # BEFORE it, the translation an English line AFTER it
    def has_word(line: str) -> bool:
        """Verbatim, or split: 离合词 get used correctly SPLIT (打交道 →
        打了三年的交道), which a plain `word in line` rejects — that threw away
        perfect examples. Characters-in-order is the fallback test; loose in
        general, but these lines were just generated FOR this word."""
        if word in line:
            return True
        it = iter(line)
        return all(ch in it for ch in word)

    def is_example(line: str) -> bool:
        return bool(HAS_CJK.search(line)) and not _mostly_ascii(line)

    # prefer a line containing the word verbatim; fall back to a split match
    ex_idx = next((i for i, l in enumerate(lines) if word in l and is_example(l)),
                  None)
    if ex_idx is None:
        ex_idx = next((i for i, l in enumerate(lines)
                       if has_word(l) and is_example(l)), None)
    resp: dict[str, str] = {"id": card_id}
    if ex_idx is not None:
        example = lines[ex_idx][:120]
        trimmed = _TRAILING_EN.sub("", example).strip()
        if trimmed and has_word(trimmed):    # keep the trim only if it leaves a valid example
            example = trimmed
        resp["example"] = example
        en = next((l[:160] for l in lines[ex_idx + 1:] if _mostly_ascii(l)), "")
        if not en:
            # the model glued or skipped the translation — a dedicated temp-0
            # call guarantees the card never lands without its English
            en = translate_user_to_english(example)
        resp["example_en"] = en
    if not gloss:
        head = lines[:ex_idx] if ex_idx is not None else lines
        resp["gloss"] = next((l[:80] for l in head if _mostly_ascii(l)), "")
    if len(resp) == 1:                       # nothing usable — keep the scraped card
        return ""
    return html.escape(json.dumps(resp, ensure_ascii=False))


# File-backed deck: localStorage is per-browser and clearable, so every deck
# change is mirrored to data/deck.json (git-ignored, lives with the repo). On
# page load, a browser with NO deck at all restores from the file — switching
# browsers or clearing site data no longer loses the cards.
DECK_FILE = c.ROOT / "data" / "deck.json"


def save_deck(deck_json: str) -> None:
    """Merge the pushed deck into the file: union by card id, incoming wins.

    Never lose cards — with LAN mode two devices push full decks, and a plain
    overwrite would clobber cards the other device added. File-only cards are
    kept; for a card both sides know, the pusher's version wins (ratings are
    last-writer-wins, an acceptable loss). Tradeoff: a card DELETED on one
    device stays in the file, so a fresh browser can resurrect it — cards are
    precious here, deletions are cheap to redo."""
    try:
        deck = json.loads(deck_json)
        if not isinstance(deck, list):
            return
    except (json.JSONDecodeError, TypeError):
        return
    try:
        existing = json.loads(DECK_FILE.read_text(encoding="utf-8")) if DECK_FILE.exists() else []
        if isinstance(existing, list):
            seen = {card.get("id") for card in deck if isinstance(card, dict)}
            deck += [card for card in existing
                     if isinstance(card, dict) and card.get("id") not in seen]
    except (json.JSONDecodeError, OSError):
        pass                                     # unreadable file — overwrite it
    DECK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DECK_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(deck, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DECK_FILE)


def deck_restore_html() -> str:
    """Contents of the deck file, re-read on every page load (the component
    value is this callable), so a fresh browser restores the latest deck."""
    try:
        if DECK_FILE.exists():
            return f'<div id="deck-file">{html.escape(DECK_FILE.read_text(encoding="utf-8"))}</div>'
    except OSError:
        pass
    return '<div id="deck-file"></div>'


def flashcards_srcdoc() -> str:
    """Embed web/flashcards.html as an <iframe srcdoc>. srcdoc iframes are
    same-origin, so the widget shares the click-to-collect localStorage deck and
    `storage` events keep it live-synced — no static-file routes needed."""
    page = (c.ROOT / "web" / "flashcards.html").read_text(encoding="utf-8")
    return f'<iframe class="cards-frame" title="flashcards" srcdoc="{html.escape(page, quote=True)}"></iframe>'


# --------------------------------------------------------------------------- #
# Reading practice: the model writes a short HSK-5 passage on a topic, rendered
# through the reading layer (pinyin + hover gloss + click-to-collect), with
# comprehension questions whose answers reveal on click. One model call/passage.
# --------------------------------------------------------------------------- #
def _reading_shell(inner: str) -> str:
    return f'<div class="reading">{inner}</div>'


def _render_passage(title: str, passage: str, translation: str, questions: list) -> str:
    """Annotated passage (collectible words + hover gloss + a 🔊), the English
    translation, and comprehension questions with reveal-on-click answers."""
    spk = (f'<button class="spk" title="朗读 · read aloud"'
           f' data-speak="{html.escape(chinese_only(passage), quote=True)}">🔊</button>')
    qs = ""
    for i, q in enumerate(questions, 1):
        qz, az = _tipped(str(q.get("q", ""))), _tipped(str(q.get("a", "")))
        qs += (f'<div class="rd-q"><div class="rd-qtext">{i}. {qz}</div>'
               f'<button class="rd-reveal" type="button">显示答案 · show answer</button>'
               f'<div class="rd-a" hidden>{az}</div></div>')
    trans = f'<div class="rd-trans">{html.escape(translation)}</div>' if translation else ""
    qblock = (f'<div class="rd-qs"><div class="rd-qs-h">理解问题 · comprehension</div>{qs}</div>'
              if qs else "")
    return (f'<div class="rd-title">{_tipped(title)} {spk}</div>'
            f'<div class="rd-passage">{_tipped(passage)}</div>{trans}{qblock}')


def gen_passage(topic: str, level_label: str = "HSK 5"):
    """Yield a placeholder, then a reading passage + comprehension questions at
    the selected HSK level on `topic` (random CONV_TOPIC when blank), rendered
    annotated."""
    level = _level_num(level_label)
    topic = (topic or "").strip() or random.choice(c.CONV_TOPICS)
    yield _reading_shell(
        f'<div class="rd-loading">正在写一篇关于“{html.escape(topic)}”的 HSK {level} 文章…<br>'
        f'writing an HSK {level} passage about “{html.escape(topic)}”…</div>')
    # The passage is built from FOUR small generations, each one simple enough
    # to be reliable:
    #   1. first half (plain text)   — asked for ~240 chars in ONE call, the
    #      fine-tuned model snaps back to its trained reply length (87–153
    #      measured across four prompt phrasings), so each half gets its own
    #      reply-sized call;
    #   2. second half (plain text)  — a big do-everything JSON call was tried
    #      and the model kept mangling the structure (questions glued outside
    #      the object), losing otherwise-good passages;
    #   3. questions (small JSON ARRAY — extract_json_array can salvage a
    #      broken closing bracket, and a failure here degrades to a passage
    #      without questions instead of no passage);
    #   4. translation of the FINAL text at temperature 0 — translated
    #      alongside the halves it drifted from what the passage actually
    #      says (user-reported).
    level_rule = (
        f"用HSK{level}或以下的词汇和语法，并尽量用上一些HSK{level}水平的词语"
        "（不要全是简单词）。不要拼音，不要换行。"
    )
    half_prompt = (
        f"为一个HSK{level}水平的学生写一篇约240字的中文阅读文章的【前半部分】，"
        f"主题是“{topic}”。\n"
        "- 前半大约120字：先引入话题，再开始讲一个具体的经历或例子，"
        "讲到一半停下（后半部分会接着写）。\n"
        f"- {level_rule}\n"
        "只返回前半部分的正文，不要标题，不要解释。"
    )
    try:
        part1 = generate([{"role": "user", "content": half_prompt}],
                         temperature=0.8, max_tokens=400).strip().strip('"“”')
        if not HAS_CJK.search(part1):
            raise ValueError("empty first half")
        finish_prompt = (
            f"下面是一篇给HSK{level}水平学生的中文阅读文章的前半部分（主题：{topic}）：\n"
            f"{part1}\n\n"
            "请写出文章的【后半部分】，大约120字：接着前半继续讲（不要重新开头），"
            f"补充细节，最后总结或给出看法。{level_rule}\n"
            "只返回后半部分的正文，不要标题，不要解释。"
        )
        part2 = generate([{"role": "user", "content": finish_prompt}],
                         temperature=0.8, max_tokens=400).strip().strip('"“”')
        passage = part1 + part2 if HAS_CJK.search(part2) else part1
        # the translation gets its OWN temperature-0 call from the FINAL joined text
        translation = translate_user_to_english(passage, max_tokens=500, max_chars=1200)
        questions = []
        try:
            q_prompt = (
                f"根据下面这篇短文，出3个中文理解问题，每个配一个简短的中文参考答案。\n\n"
                f"{passage}\n\n"
                '只返回一个JSON数组，所有字符串写在一行内：\n'
                '[{"q":"问题","a":"答案"},{"q":"…","a":"…"},{"q":"…","a":"…"}]\n'
                "不要解释，不要用markdown。"
            )
            q_out = generate([{"role": "user", "content": q_prompt}],
                             temperature=0.7, max_tokens=400)
            questions = [q for q in extract_json_array(q_out)
                         if isinstance(q, dict) and q.get("q")]
        except Exception:  # noqa: BLE001 — a passage without questions beats no passage
            pass
        yield _reading_shell(_render_passage(topic, passage, translation, questions))
    except Exception as e:  # noqa: BLE001 — a failed passage is retryable via the button
        yield _reading_shell(
            '<div class="rd-loading">这篇没写成，请再点一次“生成短文”。<br>'
            "That one didn’t generate — click “new passage” to retry. "
            f"({html.escape(str(e)[:50])})</div>")


# --------------------------------------------------------------------------- #
# Neural TTS: a 🔊 click asks the server for a Microsoft neural zh-CN voice clip
# (edge-tts, free, no key), returned as an MP3 data URI. Client plays it and
# falls back to the browser voice on any failure (offline / package missing).
# Independent of the llm — never touches _LLM_LOCK.
# --------------------------------------------------------------------------- #
# UI voice key -> edge-tts zh-CN neural voice. synth_tts derives the key from the
# voice_pick radio's label (a second input to that event); default female.
TTS_VOICES = {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"}
TTS_VOICE_DEFAULT = "female"
_TTS_CACHE: dict[tuple[str, str], str] = {}    # (voice, text) -> data URI; repeats are instant
_TTS_CACHE_MAX = 200


def _synth_bytes(text: str, voice: str) -> bytes:
    """MP3 bytes for `text` — a blocking wrapper around edge-tts's async API.
    Lazy-imports edge_tts so a missing package degrades to browser TTS, not a
    broken app import. Runs in Gradio's worker thread (no running loop), so
    asyncio.run is safe."""
    import edge_tts

    async def run() -> bytes:
        buf = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return bytes(buf)

    return asyncio.run(run())


def synth_tts(req_json: str, voice_label: str = "") -> str:
    """TTS channel: {id, text} + the voice_pick radio's label -> escaped JSON
    {id, audio: data-URI} (or {id, fail: true} so the client uses the browser
    voice). The voice comes from the voice_pick component (a second input to this
    event), which is also what makes that radio interactive."""
    try:
        req = json.loads(req_json)
        text, rid = (req.get("text") or "").strip(), req.get("id")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ""
    if not text or rid is None:
        return ""
    # an explicit voice in the request (the flashcards iframe sends its own
    # toggle's choice) wins; otherwise the chat tab's voice_pick label decides
    vkey = req.get("voice")
    if vkey not in TTS_VOICES:
        vkey = "male" if "男" in (voice_label or "") else TTS_VOICE_DEFAULT
    key = (vkey, text)
    uri = _TTS_CACHE.get(key)
    if uri is None:
        try:
            audio = _synth_bytes(text, TTS_VOICES[vkey])
            if not audio:
                raise ValueError("empty audio")
            uri = "data:audio/mpeg;base64," + base64.b64encode(audio).decode("ascii")
            if len(_TTS_CACHE) >= _TTS_CACHE_MAX:
                _TTS_CACHE.pop(next(iter(_TTS_CACHE)))     # FIFO evict oldest
            _TTS_CACHE[key] = uri
        except Exception as e:  # noqa: BLE001 — no network / missing pkg -> browser fallback
            print(f"tts: neural synth failed ({e}); client falls back to browser voice", flush=True)
            return html.escape(json.dumps({"id": rid, "fail": True}))
    return html.escape(json.dumps({"id": rid, "audio": uri}, ensure_ascii=False))


with gr.Blocks(title="HSK-5 中文 Tutor") as demo:
    gr.HTML(HEADER_HTML)
    with gr.Tab("对话 · chat"):
        chat_html = gr.HTML(render_chat([]))
        raw_state = gr.State([])
        targets_state = gr.State(None)
        targets_bar = gr.HTML("")
        # controls row 1: 模式 pills · 水平 pills · 复习 checkbox (labels + inline
        # layout come from #mode-row CSS; the voice/speed row sits below it)
        with gr.Row(elem_id="mode-row"):
            mode = gr.Radio(
                ["问答 · Q&A", "聊天 · conversation"], value="问答 · Q&A",
                show_label=False, elem_id="mode", container=False,
            )
            # HSK level: 5 is the trained level; 4/6 steer the prompts (and pick
            # their own seed lists). An event input to respond/gen_passage/the
            # starters refresh — which is also what makes the radio clickable.
            level_pick = gr.Radio(
                ["HSK 4", "HSK 5", "HSK 6"], value="HSK 5",
                show_label=False, elem_id="level", container=False, interactive=True,
            )
            review_mode = gr.Checkbox(
                False, label="复习模式 · target only due cards", elem_id="review-mode",
                container=False,
            )
        # 🔊 neural-voice pick on its OWN row — kept out of the crowded mode-row so
        # both pills stay clickable. synth_tts reads the selection (wired below);
        # applies to chat + reading (the flashcards iframe has its own TTS).
        with gr.Row(elem_id="voice-row"):
            # voice_pick feeds synth_tts (a second input to the tts event below).
            # Being an event input is what makes a Radio interactive/clickable — a
            # Radio wired to NO event renders display-only (its pills don't click),
            # and interactive=True alone does NOT fix that.
            voice_pick = gr.Radio(
                ["女声 · female", "男声 · male"], value="女声 · female",
                show_label=False, elem_id="voice-pick", container=False, interactive=True,
            )
            # Playback speed — a plain HTML range (client-only, so no Gradio event
            # wiring needed). APP_JS scales audio.playbackRate (pitch preserved) and
            # the browser voice's rate; live-adjusts a clip already playing.
            gr.HTML(
                '<div id="speed-ctl"><span class="speed-label">语速 · speed</span>'
                '<input type="range" id="tts-speed" min="0.5" max="1.5" step="0.1" value="1">'
                '<span id="speed-val">1.0×</span></div>')
        msg = gr.Textbox(
            placeholder="用中文或英文问我… (ask in Chinese or English)",
            show_label=False, submit_btn=True, elem_id="ask",
        )
        # Mirror of the client-side flashcard deck (word fronts as JSON, due-first).
        # The deck lives in localStorage, so APP_JS keeps this hidden box in sync —
        # it's how the student's own words reach pick_targets() on the server.
        deck_words = gr.Textbox("[]", elem_id="deck-words", elem_classes=["hidden-input"],
                                show_label=False, container=False)
        # Card-example channel: collect writes a request here; the model's fresh
        # example sentence lands in #card-res, which APP_JS's observer picks up.
        card_req = gr.Textbox("", elem_id="card-req", elem_classes=["hidden-input"],
                              show_label=False, container=False)
        card_res = gr.HTML("", elem_id="card-res", elem_classes=["hidden-input"])
        card_req.change(gen_card_example, card_req, card_res)
        # Neural TTS channel: a 🔊 click writes {id,text} here; synth_tts returns
        # an MP3 data URI (or {fail}) that APP_JS plays or falls back from.
        tts_req = gr.Textbox("", elem_id="tts-req", elem_classes=["hidden-input"],
                             show_label=False, container=False)
        tts_res = gr.HTML("", elem_id="tts-res", elem_classes=["hidden-input"])
        tts_req.change(synth_tts, [tts_req, voice_pick], tts_res)
        # File-backed deck: every deck change is pushed here (debounced) and
        # written to data/deck.json; #deck-file carries the file back on load.
        deck_save = gr.Textbox("", elem_id="deck-save", elem_classes=["hidden-input"],
                               show_label=False, container=False)
        deck_save.change(save_deck, deck_save, None)
        gr.HTML(deck_restore_html, elem_classes=["hidden-input"])
        # Starter refresh: the ⟲ button writes a nonce here; a fresh model-written
        # six lands in #starters-data for the observer to swap in.
        starters_req = gr.Textbox("", elem_id="starters-req", elem_classes=["hidden-input"],
                                  show_label=False, container=False)
        starters_res = gr.HTML("", elem_classes=["hidden-input"])
        starters_req.change(
            lambda _nonce, lvl: '<div id="starters-data">'
                                + html.escape(json.dumps(gen_starters(_level_num(lvl)),
                                                         ensure_ascii=False))
                                + "</div>",
            [starters_req, level_pick], starters_res)
        # Starter chips live in plain HTML; app.js shuffles the model-written
        # STARTERS injected via HEAD_HTML (SERVER_STARTERS on the JS side) and
        # wires the clicks. Python's STARTER_POOL is only the generation fallback.
        gr.HTML('<div class="starters-row" id="starters"></div>')

        msg.submit(respond, [msg, raw_state, mode, deck_words, targets_state, review_mode, level_pick],
                   [chat_html, raw_state, msg, targets_state, targets_bar])
        clear = gr.Button("清空 · clear", elem_id="clear-btn")
        clear.click(lambda: (render_chat([]), [], "", None, ""), None,
                    [chat_html, raw_state, msg, targets_state, targets_bar])
        # targets are picked once per conversation, so toggling review mode
        # resets them — the next message re-picks under the new mode instead
        # of the checkbox being a silent no-op mid-conversation
        review_mode.change(lambda: (None, ""), None, [targets_state, targets_bar])
        # same deal for the level: targets are per-conversation, so switching
        # level re-picks them from the new level's pool on the next message
        level_pick.change(lambda: (None, ""), None, [targets_state, targets_bar])
    with gr.Tab("阅读 · reading"):
        with gr.Row(elem_id="rd-controls"):
            rd_topic = gr.Textbox(
                placeholder="主题，可留空随机 · topic (blank = random)",
                show_label=False, container=False, elem_id="rd-topic", scale=5)
            rd_btn = gr.Button("生成短文 · new passage", elem_id="rd-btn", scale=1)
            # The reading tab's OWN level selector — independent of the chat tab.
            # Like every clickable Radio here, it stays interactive by being an
            # event input (to gen_passage below); a Radio wired to no event
            # renders display-only.
            rd_level_pick = gr.Radio(
                ["HSK 4", "HSK 5", "HSK 6"], value="HSK 5",
                show_label=False, elem_id="rd-level", container=False, interactive=True,
            )
        rd_out = gr.HTML(_reading_shell(
            '<div class="rd-loading">点“生成短文”，我会按你选的HSK水平写一篇文章和几个理解问题。<br>'
            'Click “new passage” for a reading at your selected HSK level, with comprehension questions.<br>'
            '每个词都能悬停看释义、点一下收藏 · hover any word for its meaning, click to save it.</div>'))
        rd_btn.click(gen_passage, [rd_topic, rd_level_pick], rd_out)
        rd_topic.submit(gen_passage, [rd_topic, rd_level_pick], rd_out)
    with gr.Tab("卡片 · flashcards"):
        gr.HTML(flashcards_srcdoc())
    with gr.Tab("词表 · word list"):
        # Rendered entirely client-side by APP_JS from the localStorage deck
        # (the deck never touches the server).
        gr.HTML('<div id="wordlist"></div>')

if __name__ == "__main__":
    # Static files for the flashcards' 写 practice mode: the hanzi-writer lib
    # and its per-character stroke data (data/strokes/, via get_strokes.py),
    # served at /gradio_api/file=<path>. Everything stays local/offline.
    gr.set_static_paths(paths=[c.ROOT / "web" / "vendor", c.ROOT / "data" / "strokes"])
    # PORT is set by dev tooling when 7860 is taken; default stays 7860.
    # LAN=1 binds to the local network so a phone/iPad on the same Wi-Fi can be
    # the touchscreen (great for 写 stroke practice; the file-backed deck means
    # a fresh device inherits your cards). Off by default — localhost only.
    # Note: the mic needs a secure context, so voice input stays a Mac feature.
    port = int(os.environ.get("PORT", "7860"))
    lan = os.environ.get("LAN") == "1"
    if lan:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))          # no packets sent — just picks the LAN interface
            ip = s.getsockname()[0]
        except OSError:
            ip = "<your-mac-ip>"
        finally:
            s.close()
        print(f"LAN mode: open http://{ip}:{port} on your phone/iPad (same Wi-Fi)", flush=True)
    # Gradio 6 takes theme/css/js at launch() (not Blocks()).
    demo.launch(
        server_name="0.0.0.0" if lan else None,
        server_port=port,
        theme=THEME,
        css=PAGE_CSS,
        head=HEAD_HTML,
    )
