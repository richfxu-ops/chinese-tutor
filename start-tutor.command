#!/bin/zsh
# HSK-5 中文 Tutor — double-click to launch.
#
# Starts the app (loading the model takes ~15s, plus a few seconds while it
# writes fresh starter prompts) and opens your browser when it's ready. If the
# tutor is already running, it just opens the browser tab. Keep this Terminal
# window around — it shows the server log; closing it stops the tutor.
#
# Tip: drag this file to the right side of your Dock for one-click launches.

cd "$(dirname "$0")" || exit 1

URL="http://localhost:${PORT:-7860}"

# already running? just open it
if curl -s -o /dev/null --max-time 1 "$URL"; then
  echo "Tutor is already running — opening $URL"
  open "$URL"
  exit 0
fi

# prefer the repo venv when present; fall back to system python3
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "启动中… loading the tutor (model + fresh starters, ~20s)"
python3 app.py &
APP_PID=$!

until curl -s -o /dev/null --max-time 1 "$URL"; do
  sleep 1
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo ""
    echo "The tutor exited before it finished starting — see the log above."
    echo "(Press any key to close this window.)"
    read -k1 -s
    exit 1
  fi
done

echo "就绪！opening $URL"
open "$URL"

# stay attached: the log streams here, and closing this window stops the tutor
wait "$APP_PID"
