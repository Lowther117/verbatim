#!/bin/bash
# ---------------------------------------------------------------------------
#  Verbatim - macOS build. Double-click this file.
#  Installs Homebrew and Python if they are missing, installs every
#  dependency, downloads the speech models, and produces dist/Verbatim.app.
#
#  Which speech model is baked in:  VERBATIM_BAKE=parakeet-v2  (default)
#  Any id from "verbatim models" works; "none" bakes only the small helper
#  models and the app downloads a speech model on first use instead.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
LOG="$HERE/build-mac-log.txt"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

# A double-clicked .command starts with a bare PATH that has no Homebrew in it.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1

echo "==============================================="
echo "  Building Verbatim for macOS"
echo "  Full log: build-mac-log.txt"
echo "==============================================="
echo

die() {
  echo
  echo "==============================================="
  echo "  BUILD FAILED: $*"
  echo "==============================================="
  echo
  echo "Press return to close."
  read -r
  exit 1
}

ensure_brew() {
  if command -v brew >/dev/null 2>&1; then return 0; fi
  echo "Homebrew is not installed. Installing it now - it will ask for your password once."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" < /dev/tty \
    || die "Homebrew would not install."
  export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
  command -v brew >/dev/null 2>&1 || die "Homebrew installed but brew is still not on the PATH."
}

brew_install() {
  ensure_brew
  echo "  brew install $*"
  brew install "$@" < /dev/null || die "brew install $* failed."
}

# Apple's /usr/bin/python3 links against system Tk 8.5, which PyInstaller
# cannot bundle: the app builds and then never opens a window. Only accept an
# interpreter whose tkinter reports 8.6 or newer - and Python 3.11 or newer,
# the oldest version the speech engine publishes wheels for. Resolve candidates with
# readlink rather than running python3, because on a clean Mac running
# python3 pops up the Xcode command line tools installer.
pick_python() {
  local c real
  if [ -n "$VERBATIM_BUILD_PYTHON" ] && [ -x "$VERBATIM_BUILD_PYTHON" ]; then
    if "$VERBATIM_BUILD_PYTHON" -c 'import tkinter,sys;sys.exit(0 if tkinter.TkVersion>=8.6 and sys.version_info>=(3,11) else 1)' 2>/dev/null; then
      echo "$VERBATIM_BUILD_PYTHON"; return 0
    fi
  fi
  for c in /opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
           /opt/homebrew/bin/python3.11 \
           /usr/local/bin/python3.14 /usr/local/bin/python3.13 /usr/local/bin/python3.12 \
           /usr/local/bin/python3.11 \
           /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.11/bin/python3; do
    [ -e "$c" ] || continue
    real="$(readlink -f "$c" 2>/dev/null || echo "$c")"
    [ -x "$real" ] || continue
    case "$real" in /usr/bin/python3) continue ;; esac
    if "$real" -c 'import tkinter,sys;sys.exit(0 if tkinter.TkVersion>=8.6 and sys.version_info>=(3,11) else 1)' 2>/dev/null; then
      echo "$real"; return 0
    fi
  done
  return 1
}

echo "[1/8] Finding a Python with a usable Tk..."
PY="$(pick_python)" || {
  echo "      None found. Installing Python and Tk through Homebrew..."
  # a named version, not plain "python": Homebrew's default moves on, and a
  # version newer than the list above would install and then not be found
  brew_install python@3.14 python-tk@3.14
  PY="$(pick_python)" || die "Even after installing python@3.14 and python-tk@3.14, no interpreter reports Tk 8.6+."
}
echo "      using $PY"
"$PY" -c 'import sys,tkinter;print("      python",sys.version.split()[0],"tk",tkinter.TkVersion)'

echo "[2/8] Creating a clean build environment..."
VENV="$HERE/.venv-build-mac"
rm -rf "$VENV"
"$PY" -m venv "$VENV" || die "Could not create the build environment."
VPY="$VENV/bin/python"

echo "[3/8] Installing dependencies (wheels only, no compiling)..."
"$VPY" -m pip install --upgrade pip wheel >/dev/null || die "pip would not upgrade."
"$VPY" -m pip install --only-binary :all: -r "$HERE/requirements.txt" || die "A dependency has no wheel for this Mac."
"$VPY" -m pip install --only-binary :all: pyinstaller || die "PyInstaller would not install."
# miniaudio plays the read-along audio gaplessly. Optional: without it the app plays
# each sentence through afplay. Its compiled half is a cffi module that imports
# _cffi_backend at C level, which PyInstaller cannot see - hence the extra flags.
MA_FLAGS=""
if "$VPY" -m pip install --only-binary :all: miniaudio; then
  MA_FLAGS="--hidden-import miniaudio --hidden-import _miniaudio --hidden-import _cffi_backend --collect-all cffi"
else
  echo "      (miniaudio did not install - read along plays through afplay instead)"
fi

echo "[4/8] Checking the code before packaging..."
"$VPY" -c "import sys; sys.path.insert(0,'.'); import sherpa_onnx, imageio_ffmpeg, numpy; import verbatimlib.ui, verbatimlib.engine, verbatimlib.export, verbatimlib.selftest; print('      imports ok')" \
  || die "The application does not import cleanly."

BAKE="${VERBATIM_BAKE:-parakeet-v2}"
BAKE_IDS="support $BAKE"
[ "$BAKE" = "none" ] && BAKE_IDS="support"
echo "[5/8] Downloading the speech models to bake in ($BAKE_IDS)..."
echo "      About 500 MB to download the first time; kept in build-models for later builds."
# --stage copies exactly the models named here into a fresh build-bake folder,
# so a model fetched for an earlier build is not baked into this one as well.
# shellcheck disable=SC2086
"$VPY" "$HERE/verbatim.py" fetch --into "$HERE/build-models" --stage "$HERE/build-bake" $BAKE_IDS \
  || die "The models would not download."

echo "[6/8] Packaging (this takes a few minutes)..."
rm -rf "$HERE/build" "$HERE/dist"
"$VPY" -m PyInstaller --noconfirm --clean --windowed \
  --name "Verbatim" \
  --osx-bundle-identifier "uk.lowther.verbatim" \
  --collect-all sherpa_onnx \
  --collect-all imageio_ffmpeg \
  --collect-all tkinterdnd2 \
  --collect-all certifi \
  --collect-submodules verbatimlib \
  --add-data "verbatimlib/data:verbatimlib/data" \
  --add-data "build-bake:models" \
  --hidden-import tkinter \
  --hidden-import tkinter.ttk \
  --hidden-import tkinter.filedialog \
  --hidden-import tkinter.messagebox \
  $MA_FLAGS \
  --exclude-module pytest \
  "$HERE/verbatim_app.py" || die "PyInstaller failed - see the log above."
[ -d "$HERE/dist/Verbatim.app" ] || die "dist/Verbatim.app was not produced."
# PyInstaller leaves a second, unbundled copy of everything beside the .app
rm -rf "$HERE/dist/Verbatim"

echo "[7/8] Clearing quarantine and signing locally..."
# Without these two an app built on Apple silicon is killed the moment it opens.
xattr -cr "$HERE/dist/Verbatim.app" || true
codesign --force --deep --sign - "$HERE/dist/Verbatim.app" || die "Ad-hoc signing failed."

echo "[8/8] Running the self-test on the built app..."
echo "      It transcribes a short two-speaker sample for real and checks every"
echo "      export format. About a minute, and it will look idle while it works."
rm -f "$HERE/dist/verbatim-selftest.txt"
"$HERE/dist/Verbatim.app/Contents/MacOS/Verbatim" selftest || true
sleep 2
if [ -f "$HERE/dist/verbatim-selftest.txt" ]; then
  if grep -q "PROBLEMS FOUND" "$HERE/dist/verbatim-selftest.txt"; then
    echo
    echo "==============================================="
    echo "  BUILT, BUT THE SELF-TEST REPORTED PROBLEMS."
    echo "  See dist/verbatim-selftest.txt (printed above)."
    echo "==============================================="
    echo
    echo "Press return to close."
    read -r
    exit 2
  fi
else
  echo "      (no self-test report was written - check it by hand)"
fi

echo
echo "==============================================="
echo "  DONE.  dist/Verbatim.app is ready."
echo "  Drag it to your Applications folder."
echo "==============================================="
echo
echo "Press return to close."
read -r
