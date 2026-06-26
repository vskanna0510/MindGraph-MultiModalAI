#!/usr/bin/env bash
# Activate virtual environment (source this file)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
echo "Activated MindGraph++ virtual environment"
