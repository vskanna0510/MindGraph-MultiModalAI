# MindGraph++ Setup Script (Linux / WSL2)
# Usage: chmod +x scripts/setup.sh && ./scripts/setup.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> MindGraph++ Setup (Linux/WSL2)"

command -v python3 >/dev/null 2>&1 || { echo "Python 3.11+ required"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

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
  echo "Created .env from .env.example"
fi

mkdir -p backend/logs assets/uploads ml_pipeline/weights datasets/raw

if command -v flutter >/dev/null 2>&1 && [ -d "flutter_app" ]; then
  cd flutter_app && flutter pub get && cd ..
fi

docker compose up -d

echo "==> Validation"
python -c "from app.config.settings import get_settings; print('Settings OK')"
cd backend && pytest tests/unit/test_health.py -q

echo "==> Setup complete"
echo "Run: source .venv/bin/activate && make backend"
