#!/usr/bin/env bash
# MindGraph++ Setup Script (macOS)
# Usage: chmod +x scripts/setup_mac.sh && ./scripts/setup_mac.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> MindGraph++ Setup (macOS)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Install Python 3.11+ via Homebrew: brew install python@3.11"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Install Docker Desktop for Mac"
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-test.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
fi

mkdir -p backend/logs assets/uploads ml_pipeline/weights datasets/raw

if command -v flutter >/dev/null 2>&1 && [ -d "flutter_app" ]; then
  (cd flutter_app && flutter pub get)
fi

docker compose up -d

PYTHONPATH=backend python -c "from app.config.settings import get_settings; print('Settings OK')"
(cd backend && pytest tests/unit/test_health.py -q)

echo "==> Setup complete. Run: make backend"
