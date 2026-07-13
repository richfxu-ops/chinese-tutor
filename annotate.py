"""Reading layer: turn the tutor's Chinese into HTML with pinyin over every
character and a hover gloss on every word.

The model never sees any of this — it outputs clean bilingual text, and we
annotate deterministically here:
  - pypinyin  → pinyin reading per character (always correct, no model guessing)
  - jieba     → word segmentation (Chinese has no spaces)
  - CC-CEDICT → English gloss per word, shown via the native `title` tooltip

Only CJK runs are annotated; English (and punctuation) pass through untouched, so
a bilingual answer renders correctly with zero special-casing.

CC-CEDICT is optional: if the dict file is missing, pinyin still works and the
hover just shows the word's pinyin. Run `python get_cedict.py` to fetch it.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import jieba
from pypinyin import Style, pinyin

CEDICT_FILE = Path(__file__).resolve().parent / "cedict_ts.u8"

# One or more CJK ideographs in a row. HAS_CJK is public: app.py shares it
# (TTS text extraction, correction-chip gating) so the two files can't drift
# on what counts as Chinese.
_CJK_RUN = re.compile(r"[一-鿿]+")
HAS_CJK = re.compile(r"[一-鿿]")


class Entry(NamedTuple):
    """One CC-CEDICT line for a simplified form."""
    key: str          # normalized pinyin (lowercase, no spaces, u:→v) — the shape _match_key produces
    is_proper: bool   # CEDICT reading was capitalized (钱 Qian2 "surname") — loses ties in selection
    glosses: list[str]
    reading: str      # original spaced reading, e.g. "di4 dao5"


@lru_cache(maxsize=1)
def load_cedict() -> dict[str, list[Entry]]:
    """Parse CC-CEDICT into {simplified: [entries]}.

    ALL entries per form are kept (heteronyms like 还 hái/huán are separate
    CEDICT lines); _pick_glosses matches them against the word's in-context
    reading. The pinyin key is normalized to lowercase, no spaces, u:→v — the
    same shape pypinyin's TONE3 style produces. is_proper marks entries whose
    CEDICT reading is capitalized (钱 Qian2 "surname Qian"): lowercasing makes
    their key collide with the common word, so selection demotes them.
    Empty if the file is absent.
    """
    if not CEDICT_FILE.exists():
        return {}
    out: dict[str, list[Entry]] = {}
    for line in CEDICT_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or " " not in line:
            continue
        # format: traditional simplified [pin1 yin1] /gloss1/gloss2/
        try:
            simp = line.split(" ", 2)[1]
            reading = line[line.index("[") + 1: line.index("]")]
            glosses = line[line.index("/") + 1: line.rindex("/")].split("/")
        except (IndexError, ValueError):
            continue
        key = reading.lower().replace(" ", "").replace("u:", "v")
        out.setdefault(simp, []).append(Entry(key, reading != reading.lower(), glosses, reading))
    return out


def _word_pinyin(word: str) -> list[str]:
    """Per-character pinyin with tone marks, using word context for correct readings."""
    return [syl[0] for syl in pinyin(word, style=Style.TONE)]


def _match_key(word: str) -> str:
    """The word's in-context reading in CEDICT-key shape (hai2mei2, lv4...)."""
    syls = pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
    return "".join(s[0] for s in syls).lower()


# Senses that are metadata rather than meanings — dropped when real senses exist.
_NON_SENSE = ("CL:", "variant of", "old variant of", "used in", "also pr.", "see ")


def _clean_senses(glosses: list[str]) -> list[str]:
    return [g for g in glosses if not g.startswith(_NON_SENSE) and len(g) <= 80]


# Tone-number pinyin → tone marks ("dao4" → "dào", "lv3" → "lǚ"; tone 5 bare).
_TONE_MARKS = {"a": "āáǎà", "o": "ōóǒò", "e": "ēéěè", "i": "īíǐì", "u": "ūúǔù", "ü": "ǖǘǚǜ"}


def _mark_syllable(syl: str) -> str:
    m = re.fullmatch(r"([a-zü]+)([1-5])?", syl.lower().replace("u:", "ü").replace("v", "ü"))
    if not m:
        return syl
    body, tone = m.group(1), int(m.group(2) or 5)
    if tone == 5:
        return body
    for target in "aeo":                      # mark a, else e, else o...
        i = body.find(target)
        if i >= 0:
            break
    else:                                     # ...else the last of i/u/ü (iu→u, ui→i)
        idxs = [j for j, ch in enumerate(body) if ch in "iuü"]
        if not idxs:
            return body
        i = idxs[-1]
    return body[:i] + _TONE_MARKS[body[i]][tone - 1] + body[i + 1:]


def _pick_glosses(word: str, cedict: dict[str, list[Entry]]) -> str | None:
    """Glosses for the CEDICT entry matching the word's in-context pinyin.

    pypinyin picks the reading from its phrase data (还钱 → huán, 还没 → hái),
    so the tooltip's senses agree with the ruby pinyin above the characters.
    Proper-noun entries lose ties (钱: money, not the surname). No pinyin match
    (a single-char token read differently than CEDICT's default) falls back to
    the first common-word entry. Long example-laden senses are dropped when
    shorter ones exist, so the tooltip stays a tooltip.
    """
    entries = cedict.get(word)
    if not entries:
        return None
    key = _match_key(word)
    pool = [e for e in entries if e.key == key] or entries
    pool = [e for e in pool if not e.is_proper] or pool
    # merge senses across same-reading entries (间 jiān: between AND room),
    # 4 from the primary entry + 2 from each further one, 6 senses max; entries
    # with only metadata senses are skipped, not resurrected
    picked: list[str] = []
    for i, entry in enumerate(pool):
        real = _clean_senses(entry.glosses)
        picked += [g for g in real if g not in picked][: 4 if i == 0 else 2]
        if len(picked) >= 6:
            break
    if not picked:                                     # every sense was metadata
        picked = pool[0].glosses
    return "; ".join(picked[:6])


# --------------------------------------------------------------------------- #
# Sense disambiguation hooks (the model call itself lives in app.py)
# --------------------------------------------------------------------------- #
# Words whose in-context reading is basically always right — not worth a model
# round-trip even though CEDICT has several entries for them.
_SKIP_DISAMBIG = {"的", "了"}


def _meaningful_entries(word: str) -> list[Entry]:
    cedict = load_cedict()
    return [e for e in cedict.get(word, []) if not e.is_proper and _clean_senses(e.glosses)]


def ambiguous_words(text: str) -> list[str]:
    """Annotation tokens of `text` with 2+ real CEDICT entries (地道, 还, 花...),
    in first-appearance order — the words worth asking the model about."""
    out: list[str] = []
    for m in _CJK_RUN.finditer(text):
        for token in jieba.cut(m.group()):
            if (
                token not in out
                and token not in _SKIP_DISAMBIG
                and HAS_CJK.search(token)
                and len(_meaningful_entries(token)) >= 2
            ):
                out.append(token)
    return out


def unglossed_words(text: str) -> list[str]:
    """Annotation tokens of `text` with NO CEDICT entry at all (jieba compounds
    like 一只/很累) — their tooltips would be pinyin-only, so the per-turn model
    call writes them a definition."""
    cedict = load_cedict()
    out: list[str] = []
    for m in _CJK_RUN.finditer(text):
        for token in jieba.cut(m.group()):
            if token not in out and HAS_CJK.search(token) and token not in cedict:
                out.append(token)
    return out


def sense_options(word: str) -> list[dict]:
    """One dict per candidate sense of `word`, for the disambiguation prompt and
    for building the override once the model picks: {reading, syllables, brief, gloss}."""
    opts = []
    for entry in _meaningful_entries(word):
        senses = _clean_senses(entry.glosses)
        syllables = [_mark_syllable(s) for s in entry.reading.split()]
        opts.append({
            "reading": " ".join(syllables),
            "syllables": syllables,
            "brief": "; ".join(senses[:2]),
            "gloss": "; ".join(senses[:4]),
        })
    return opts


def _annotate_word(word: str, cedict: dict[str, list[Entry]],
                   overrides: dict[str, tuple[list[str], str]] | None) -> str:
    """One CJK word → <span title=...><ruby>字<rt>zì</rt></ruby>...</span>.

    An override — (syllables, gloss) picked by the model for this message —
    replaces both the gloss and, when the syllable count lines up, the ruby
    pinyin (地道 as "authentic" reads dì dao, not dì dào)."""
    syllables = _word_pinyin(word)
    gloss = _pick_glosses(word, cedict)
    if overrides and word in overrides:
        ov_syls, gloss = overrides[word]
        if len(ov_syls) == len(word):
            syllables = ov_syls
    ruby = "".join(
        f"<ruby>{html.escape(ch)}<rt>{html.escape(py)}</rt></ruby>"
        for ch, py in zip(word, syllables)
    )
    reading = " ".join(syllables)
    title = f"{reading} — {gloss}" if gloss else reading
    return f'<span class="hz" title="{html.escape(title, quote=True)}">{ruby}</span>'


def _annotate_cjk_run(run: str, cedict: dict[str, list[Entry]],
                      overrides: dict[str, tuple[list[str], str]] | None) -> str:
    """Segment a CJK run and annotate each token (punctuation passes through)."""
    parts = []
    for token in jieba.cut(run):
        if HAS_CJK.search(token):
            parts.append(_annotate_word(token, cedict, overrides))
        else:
            parts.append(html.escape(token))
    return "".join(parts)


def annotate(text: str, overrides: dict[str, tuple[list[str], str]] | None = None) -> str:
    """Annotate all CJK spans in `text`; leave non-CJK (English) escaped but plain.

    `overrides` maps word → (syllables, gloss) for senses the model picked in
    this message's context (see app.py's disambiguate). Newlines become <br>
    so multi-line tutor replies keep their shape in HTML.
    """
    cedict = load_cedict()
    out, last = [], 0
    for m in _CJK_RUN.finditer(text):
        out.append(html.escape(text[last:m.start()]))                 # non-CJK gap
        out.append(_annotate_cjk_run(m.group(), cedict, overrides))   # CJK run
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


if __name__ == "__main__":
    # quick manual check
    sample = "既然你累了，就早点休息吧。\nSince you're tired, get some rest."
    print(annotate(sample))
