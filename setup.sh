#!/bin/bash
# One-time setup for the HSK-5 tutor. From the repo root:
#
#     ./setup.sh
#
# Creates a virtualenv, installs the serve deps, fetches the dictionary and
# stroke data, and downloads the fine-tuned model (~9 GB) from Hugging Face.
# Safe to re-run: every step skips work that's already done, and the model
# download RESUMES if interrupted. When it finishes, start the tutor with
# ./start-tutor.command (or double-click it in Finder).
set -euo pipefail
cd "$(dirname "$0")"

# Where the model lives (override with HF_REPO=user/repo ./setup.sh)
HF_REPO="${HF_REPO:-richfxu-ops/hsk5-tutor-14b-gguf}"
GGUF="outputs/hsk5-tutor-q4_k_m.gguf"
GGUF_URL="https://huggingface.co/${HF_REPO}/resolve/main/hsk5-tutor-q4_k_m.gguf"
MIN_GGUF_BYTES=8000000000   # a complete 14B Q4 is ~9 GB; anything smaller is a truncated download

# ---- sanity: this app is built for Apple Silicon ---------------------------
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "⚠️  Built for Apple Silicon Macs (llama.cpp Metal). Other platforms need"
  echo "   a manual llama-cpp-python install; continuing anyway…"
fi
ram_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
if [ "$ram_gb" -gt 0 ] && [ "$ram_gb" -lt 16 ]; then
  echo "⚠️  ${ram_gb} GB RAM detected — the 14B model wants 16 GB+; expect heavy swapping."
fi

# ---- python venv + serve deps ----------------------------------------------
command -v python3 >/dev/null || {
  echo "❌ python3 not found — install it (python.org or Homebrew), then re-run."; exit 1; }
[ -d .venv ] || python3 -m venv .venv
echo "Installing dependencies…"
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements-app.txt

# ---- dictionary + stroke data (fetched once, git-ignored) -------------------
[ -f cedict_ts.u8 ]  || { echo "Fetching CC-CEDICT (hover glosses)…"; ./.venv/bin/python get_cedict.py; }
[ -d data/strokes ]  || { echo "Fetching stroke data (写 practice, ~46 MB)…"; ./.venv/bin/python get_strokes.py; }

# ---- the model (~9 GB, resumable) --------------------------------------------
mkdir -p outputs
if [ ! -f "$GGUF" ] || [ "$(stat -f%z "$GGUF" 2>/dev/null || stat -c%s "$GGUF" 2>/dev/null || echo 0)" -lt "$MIN_GGUF_BYTES" ]; then
  echo "Downloading the model (~9 GB) — interruptions are fine, re-running resumes…"
  curl -L -C - --fail --progress-bar -o "$GGUF" "$GGUF_URL"
fi
# a truncated model file fails confusingly at load time — check the size now
size=$(stat -f%z "$GGUF" 2>/dev/null || stat -c%s "$GGUF")
if [ "$size" -lt "$MIN_GGUF_BYTES" ]; then
  echo "❌ $GGUF is only $((size / 1000000)) MB — the download looks incomplete."
  echo "   Re-run ./setup.sh to resume it."
  exit 1
fi

echo ""
echo "✅ Setup complete — start the tutor with:  ./start-tutor.command"
echo "   (first launch takes ~20s: model load + fresh starter prompts)"
