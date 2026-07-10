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
def load_cedict() -> dict[str, str]:
    """Parse CC-CEDICT into {simplified: 'gloss; gloss'}. Empty if the file is absent."""
    if not CEDICT_FILE.exists():
        return {}
    out: dict[str, str] = {}
    for line in CEDICT_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or " " not in line:
            continue
        # format: traditional simplified [pin1 yin1] /gloss1/gloss2/
        try:
            simp = line.split(" ", 2)[1]
            glosses = line[line.index("/") + 1: line.rindex("/")]
        except (IndexError, ValueError):
            continue
        out.setdefault(simp, "; ".join(glosses.split("/")[:3]))
    return out


def _word_pinyin(word: str) -> list[str]:
    """Per-character pinyin with tone marks, using word context for correct readings."""
    return [syl[0] for syl in pinyin(word, style=Style.TONE)]


def _annotate_word(word: str, cedict: dict[str, str]) -> str:
    """One CJK word → <span title=...><ruby>字<rt>zì</rt></ruby>...</span>."""
    syllables = _word_pinyin(word)
    ruby = "".join(
        f"<ruby>{html.escape(ch)}<rt>{html.escape(py)}</rt></ruby>"
        for ch, py in zip(word, syllables)
    )
    reading = " ".join(syllables)
    gloss = cedict.get(word)
    title = f"{reading} — {gloss}" if gloss else reading
    return f'<span class="hz" title="{html.escape(title, quote=True)}">{ruby}</span>'


def _annotate_cjk_run(run: str, cedict: dict[str, str]) -> str:
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
