#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
#  FlashGuard — Quick Start
# ──────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ⚡ FLASHGUARD — Flash Crash Prediction Server     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found. Please install Python 3.10+."
  exit 1
fi

# Install / update deps quietly
echo "▶  Installing dependencies…"
pip install -q -r requirements.txt

echo "▶  Starting API server…"
echo "▶  Open http://localhost:5000 in your browser"
echo ""
python3 api_server.py
