"""Fetch Hanzi Writer stroke data — one JSON per character — for offline use.

The flashcards' 写 practice mode animates and quizzes stroke order via
hanzi-writer (web/vendor/hanzi-writer.min.js, committed). The per-character
stroke data (~9k files, ~46MB) is too big for git, so like cedict_ts.u8 it is
fetched once into data/strokes/ and git-ignored.

    python get_strokes.py
"""

from __future__ import annotations

import io
import ssl
import tarfile
import urllib.request
from pathlib import Path

URL = "https://registry.npmjs.org/hanzi-writer-data/-/hanzi-writer-data-2.0.1.tgz"
OUT = Path(__file__).resolve().parent / "data" / "strokes"


def _ssl_context() -> ssl.SSLContext:  # duplicated in get_cedict.py — the fetch scripts stay standalone
    """Prefer certifi's CA bundle — python.org macOS builds often lack system certs."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} ...")
    buf = io.BytesIO(urllib.request.urlopen(URL, context=_ssl_context()).read())
    n = 0
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            # keep only the per-character files (字.json), not package.json/all.json
            if m.isfile() and name.endswith(".json") and len(name.removesuffix(".json")) == 1:
                (OUT / name).write_bytes(tf.extractfile(m).read())
                n += 1
    print(f"wrote {n} character files -> {OUT}")


if __name__ == "__main__":
    main()
