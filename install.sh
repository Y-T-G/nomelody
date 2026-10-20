#!/usr/bin/env bash
set -e

BIN_DIR="${PREFIX:-/usr/local}/bin"
[ -d "$HOME/bin" ] && BIN_DIR="$HOME/bin"

mkdir -p "$BIN_DIR"
chmod +x "$(dirname "$0")/nomelody.py"
cp "$(dirname "$0")/nomelody.py" "$BIN_DIR/nomelody"
ln -sf "$BIN_DIR/nomelody" "$BIN_DIR/vocal-isolate"

echo "Installed nomelody into $BIN_DIR/nomelody"
