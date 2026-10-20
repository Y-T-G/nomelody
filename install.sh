#!/usr/bin/env bash
set -e

BIN_DIR="${PREFIX:-/usr/local}/bin"
[ -d "$HOME/bin" ] && BIN_DIR="$HOME/bin"

mkdir -p "$BIN_DIR"
chmod +x "$(dirname "$0")/vocal_isolate.py"
cp "$(dirname "$0")/vocal_isolate.py" "$BIN_DIR/nomelody"
ln -sf "$BIN_DIR/nomelody" "$BIN_DIR/vocal-isolate"

echo "Installed nomelody (and alias vocal-isolate) into $BIN_DIR/"
