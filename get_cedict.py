"""Download the CC-CEDICT dictionary for the reading layer's hover glosses.

CC-CEDICT is a free Chinese-English dictionary (CC-BY-SA 4.0). We don't commit
it (it's a data asset that updates independently); fetch it once:

    python get_cedict.py

Attribution (also in the README): CC-CEDICT, https://www.mdbg.net/chinese/dictionary?page=cc-cedict — CC-BY-SA 4.0.
"""

from __future__ import annotations

import gzip
import ssl
import urllib.request
from pathlib import Path

URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
OUT = Path(__file__).resolve().parent / "cedict_ts.u8"


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle — python.org macOS builds often lack system certs.
    (Duplicated in get_strokes.py: the fetch scripts stay standalone on purpose.)"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def main() -> None:
    if OUT.exists():
        print(f"already present: {OUT} ({OUT.stat().st_size // 1024} KB)")
        return
    print(f"downloading CC-CEDICT from {URL} ...")
    with urllib.request.urlopen(URL, context=_ssl_context()) as resp:   # noqa: S310 — known static URL
        data = gzip.decompress(resp.read())
    OUT.write_bytes(data)
    entries = sum(1 for line in data.decode("utf-8").splitlines() if not line.startswith("#"))
    print(f"saved {OUT} — {entries} entries")


if __name__ == "__main__":
    main()
