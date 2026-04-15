#!/usr/bin/env bash
# ═══════════════════════════════════════════════════
#  NeuroGuard — Start Backend Server
#  Usage: bash run_backend.sh
# ═══════════════════════════════════════════════════

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "  ⬡  NeuroGuard — Neuromorphic Driver Monitoring"
echo "  ════════════════════════════════════════════════"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "  ❌  python3 not found. Install Python 3.9+."
  exit 1
fi

echo "  🔍  Checking dependencies..."
pip3 install -q flask flask-cors opencv-python numpy 2>/dev/null || true
echo "  ✅  Core dependencies ready."

# Model status
MODEL_PATH="$PROJECT_ROOT/saved_models/model.pth"
ROOT_MODEL="$PROJECT_ROOT/model.pth"
if [ -f "$MODEL_PATH" ] || [ -f "$ROOT_MODEL" ]; then
  echo "  ✅  Trained model found — running in SNN mode."
else
  echo "  ⚠️   No model.pth found — running in heuristic (OpenCV) mode."
  echo "       Train first: python training/train.py"
  echo "       Then move model.pth → saved_models/model.pth"
fi

echo ""
echo "  🚀  Starting API server on http://localhost:5050"
echo "  📁  Open frontend: frontend/index.html in your browser"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

python3 -m backend.app
