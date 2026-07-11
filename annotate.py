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

import jieba
from pypinyin import Style, pinyin

CEDICT_FILE = Path(__file__).resolve().parent / "cedict_ts.u8"

# One or more CJK ideographs in a row.
_CJK_RUN = re.compile(r"[一-鿿]+")
_HAS_CJK = re.compile(r"[一-鿿]")


@lru_cache(maxsize=1)
def load_cedict() -> dict[str, list[tuple[str, bool, list[str]]]]:
    """Parse CC-CEDICT into {simplified: [(pinyin key, is_proper, [glosses])]}.

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
    out: dict[str, list[tuple[str, bool, list[str]]]] = {}
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
        out.setdefault(simp, []).append((key, reading != reading.lower(), glosses))
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


def _pick_glosses(word: str, cedict: dict[str, list[tuple[str, bool, list[str]]]]) -> str | None:
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
    pool = [e for e in entries if e[0] == key] or entries
    pool = [e for e in pool if not e[1]] or pool       # demote proper nouns
    # merge senses across same-reading entries (间 jiān: between AND room),
    # 4 from the primary entry + 2 from each further one, 6 senses max; entries
    # with only metadata senses are skipped, not resurrected
    picked: list[str] = []
    for i, (_, _, glosses) in enumerate(pool):
        real = [g for g in glosses if not g.startswith(_NON_SENSE) and len(g) <= 80]
        picked += [g for g in real if g not in picked][: 4 if i == 0 else 2]
        if len(picked) >= 6:
            break
    if not picked:                                     # every sense was metadata
        picked = pool[0][2]
    return "; ".join(picked[:6])


def _annotate_word(word: str, cedict: dict[str, list[tuple[str, list[str]]]]) -> str:
    """One CJK word → <span title=...><ruby>字<rt>zì</rt></ruby>...</span>."""
    syllables = _word_pinyin(word)
    ruby = "".join(
        f"<ruby>{html.escape(ch)}<rt>{html.escape(py)}</rt></ruby>"
        for ch, py in zip(word, syllables)
    )
    reading = " ".join(syllables)
    gloss = _pick_glosses(word, cedict)
    title = f"{reading} — {gloss}" if gloss else reading
    return f'<span class="hz" title="{html.escape(title, quote=True)}">{ruby}</span>'


def _annotate_cjk_run(run: str, cedict: dict[str, list[tuple[str, list[str]]]]) -> str:
    """Segment a CJK run and annotate each token (punctuation passes through)."""
    parts = []
    for token in jieba.cut(run):
        if _HAS_CJK.search(token):
            parts.append(_annotate_word(token, cedict))
        else:
            parts.append(html.escape(token))
    return "".join(parts)


def annotate(text: str) -> str:
    """Annotate all CJK spans in `text`; leave non-CJK (English) escaped but plain.

    Newlines become <br> so multi-line tutor replies keep their shape in HTML.
    """
    cedict = load_cedict()
    out, last = [], 0
    for m in _CJK_RUN.finditer(text):
        out.append(html.escape(text[last:m.start()]))      # non-CJK gap
        out.append(_annotate_cjk_run(m.group(), cedict))   # CJK run
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


if __name__ == "__main__":
    # quick manual check
    sample = "既然你累了，就早点休息吧。\nSince you're tired, get some rest."
    print(annotate(sample))
