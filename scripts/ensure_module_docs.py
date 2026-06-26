#!/usr/bin/env python3
"""Ensure every project directory contains a README.md.

Purpose:
    Enforce Master Prompt 1 Part 2 documentation standards across the monorepo.

Usage:
    python scripts/ensure_module_docs.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "node_modules",
    "__pycache__",
    ".dart_tool",
}

DOC_SET = (
    "architecture.md",
    "workflow.md",
    "limitations.md",
    "future_work.md",
)

MODULE_ROOTS = (
    "backend",
    "flutter_app",
    "ml_pipeline",
    "deployment",
    "monitoring",
    "benchmarking",
    "graphs",
    "datasets",
    "experiments",
    "configs",
    "docs",
    "scripts",
    "tests",
    "tools",
    "assets",
    "papers",
    "thesis",
    "reports",
)


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def title_from_path(path: Path) -> str:
    name = path.name.replace("_", " ").replace("-", " ")
    return name.title() if name else "MindGraph++"


def readme_content(path: Path) -> str:
    relative = path.relative_to(ROOT)
    return (
        f"# {title_from_path(path)}\n\n"
        f"Module path: `{relative}`\n\n"
        "See parent documentation for architecture and workflows.\n"
    )


def module_doc_content(path: Path, doc_name: str) -> str:
    title = doc_name.replace(".md", "").replace("_", " ").title()
    relative = path.relative_to(ROOT)
    return (
        f"# {title_from_path(path)} — {title}\n\n"
        f"Module: `{relative}`\n\n"
        f"Document `{doc_name}` for the {relative} module.\n"
    )


def ensure_readmes() -> int:
    created = 0
    for directory in sorted(ROOT.rglob("*")):
        if not directory.is_dir() or should_skip(directory):
            continue
        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(readme_content(directory), encoding="utf-8")
            created += 1
    return created


def ensure_module_docs() -> int:
    created = 0
    for module in MODULE_ROOTS:
        module_path = ROOT / module
        if not module_path.is_dir():
            continue
        for doc_name in DOC_SET:
            doc_path = module_path / doc_name
            if not doc_path.exists():
                doc_path.write_text(module_doc_content(module_path, doc_name), encoding="utf-8")
                created += 1
    return created


def main() -> None:
    readme_count = ensure_readmes()
    doc_count = ensure_module_docs()
    print(f"Created {readme_count} README.md files and {doc_count} module docs.")


if __name__ == "__main__":
    main()
