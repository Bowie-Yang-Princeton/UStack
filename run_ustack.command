#!/bin/bash
# UStack launcher for a Mac: double-click to run UStack from these sources.
#
# If the Finder refuses to open it: right-click (Control-click) it › Open › Open, or run it from the
# Terminal with:   bash "<this folder>/run_ustack.command"
#
# The first start makes a private Python environment in ~/Library/Caches/UStack-build (outside
# Dropbox) from a Python 3.10-3.14 found on this Mac (python.org, Homebrew, Anaconda, or the one
# UAnalysis already uses; USTACK_PYTHON=/path/to/python3 picks one) and installs requirements.txt
# into it - a few minutes, once. Later starts are quick. `--fresh` makes the environment anew.
set -uo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
WORK="$HOME/Library/Caches/UStack-build"
ARCH="$(uname -m)"
if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then ARCH="arm64"; fi
VENV="$WORK/venv-$ARCH"
PY="$VENV/bin/python"
READY="$VENV/.ustack-ready"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() {
    printf '\n\033[31mError:\033[0m %s\n' "$*" >&2
    read -n 1 -s -r -p "Press any key to close."
    echo
    exit 1
}

find_python() {
    local candidates=() exe
    [[ -n "${USTACK_PYTHON:-}" ]] && candidates+=("$USTACK_PYTHON")
    for v in 3.13 3.12 3.11 3.14 3.10; do candidates+=("/Library/Frameworks/Python.framework/Versions/$v/bin/python3"); done
    candidates+=(/opt/homebrew/bin/python3 /usr/local/bin/python3 python3.13 python3.12 python3.11 python3.10 python3
                 "$HOME/Library/Caches/UAnalysis-build/venv-$ARCH/bin/python"
                 "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3" /opt/anaconda3/bin/python3 /opt/miniconda3/bin/python3)
    for c in "${candidates[@]}"; do
        exe="$(command -v "$c" 2>/dev/null)" || continue
        [[ "$exe" == "/usr/bin/python3" ]] && continue  # Apple's stub (Python 3.9; may ask to install developer tools)
        "$exe" -c "import sys, venv, platform; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 14) and platform.machine() == '$ARCH'))" 2>/dev/null || continue
        echo "$exe"
        return 0
    done
    return 1
}

if [[ "${1:-}" == "--fresh" ]]; then
    shift
    rm -rf "$VENV"
fi

if [[ ! -f "$READY" ]]; then
    BASE="$(find_python)" || fail "No Python 3.10-3.14 for this Mac ($ARCH) was found.
Install one from https://www.python.org/downloads/macos/ (the macOS 64-bit universal2 installer),
or point to one with USTACK_PYTHON=/path/to/python3, and run this again."
    say "Making UStack's Python environment in $VENV from $BASE ($("$BASE" -V 2>&1)) - first time only, a few minutes"
    mkdir -p "$WORK"
    rm -rf "$VENV"
    "$BASE" -m venv "$VENV" || fail "Could not create the environment in $VENV."
    "$PY" -m pip install --disable-pip-version-check --quiet --upgrade pip
    "$PY" -m pip install --disable-pip-version-check -r "$PROJECT/requirements.txt" || fail "Installing the requirements failed (see above)."
    "$PY" -c "from PyQt6 import QtCore, QtSvg, QtWidgets; import numpy, cv2, klayout.db" || fail "The packages do not import in $VENV."
    touch "$READY"
fi

cd "$PROJECT" || fail "Cannot open $PROJECT"
export PYTHONDONTWRITEBYTECODE=1  # no __pycache__ in the Dropbox folder
say "Starting UStack ..."
if ! "$PY" -m ustack "$@"; then
    fail "UStack stopped with an error (see the messages above)."
fi
