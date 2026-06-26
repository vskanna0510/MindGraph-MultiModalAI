#!/usr/bin/env python3
"""Validate MindGraph++ development environment."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COMMANDS = ("python", "docker", "git")
OPTIONAL_COMMANDS = ("flutter", "node", "ffmpeg")


def check_command(name: str) -> bool:
    return shutil.which(name) is not None


def check_python_version() -> tuple[bool, str]:
    version = sys.version_info
    ok = version >= (3, 11)
    return ok, f"{version.major}.{version.minor}.{version.micro}"


def main() -> int:
    print("MindGraph++ Environment Validation")
    print("=" * 40)

    py_ok, py_version = check_python_version()
    print(f"Python {py_version}: {'OK' if py_ok else 'FAIL (need 3.11+)'}")
    if not py_ok:
        return 1

    for cmd in REQUIRED_COMMANDS:
        status = "OK" if check_command(cmd) else "MISSING"
        print(f"{cmd}: {status}")
        if status == "MISSING" and cmd != "python":
            return 1

    for cmd in OPTIONAL_COMMANDS:
        status = "OK" if check_command(cmd) else "optional (not installed)"
        print(f"{cmd}: {status}")

    venv = ROOT / ".venv"
    print(f".venv: {'OK' if venv.exists() else 'MISSING — run scripts/devops/create_env'}")
    print(f".env: {'OK' if (ROOT / '.env').exists() else 'MISSING — copy .env.example'}")

    if shutil.which("docker"):
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"docker compose: {'OK' if result.returncode == 0 else 'FAIL'}")

    print("=" * 40)
    print("Validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
