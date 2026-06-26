#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade -r requirements-dev.txt
pip install --upgrade -r requirements-prod.txt
echo "Dependencies upgraded."
