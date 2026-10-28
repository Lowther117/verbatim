#!/bin/bash
# Run Verbatim from source on macOS, setting up a virtual environment the
# first time. Use build-app.command if you want a standalone .app instead.
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
VENV=".venv"
pick() {
  for c in /opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
           /opt/homebrew/bin/python3.11 \
           /usr/local/bin/python3.14 /usr/local/bin/python3.13 /usr/local/bin/python3.12 \
           /usr/local/bin/python3.11 \
           /Library/Frameworks/Python.framework/Versions/3.1*/bin/python3; do
    [ -x "$c" ] || continue
    if "$c" -c 'import tkinter,sys; sys.exit(0 if tkinter.TkVersion>=8.6 and sys.version_info>=(3,11) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}
if [ ! -x "$VENV/bin/python" ]; then
  echo "First run - setting Python up. This happens once."
  PY="$(pick)" || { echo "No Python 3.11+ with a usable Tk found. Run build-app.command once; it installs one."; read -r; exit 1; }
  "$PY" -m venv "$VENV" || { echo "Could not create the environment."; read -r; exit 1; }
fi
# The environment can exist with its packages missing (a first run whose pip
# install failed), so check they import rather than trusting the folder.
if ! "$VENV/bin/python" -c 'import sherpa_onnx, numpy, imageio_ffmpeg, certifi' >/dev/null 2>&1; then
  echo "Installing the packages..."
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
  "$VENV/bin/python" -m pip install --only-binary :all: -r requirements.txt || { read -r; exit 1; }
fi
# miniaudio (gapless read-along audio) is optional and may have no wheel yet for a
# brand-new Python, so it is tried once rather than on every launch.
if [ ! -e "$VENV/.miniaudio-tried" ]; then
  "$VENV/bin/python" -m pip install --only-binary :all: miniaudio >/dev/null 2>&1 \
    || echo "(miniaudio did not install - read along plays through afplay instead)"
  touch "$VENV/.miniaudio-tried"
fi
exec "$VENV/bin/python" verbatim.py "$@"
