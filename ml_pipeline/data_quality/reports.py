"""Automated report generation."""

from __future__ import annotations

from pathlib import Path

from ml_pipeline.data_quality.types import DataQualityResult, DatasetPassport


def generate_master_report(result: DataQualityResult, passport: DatasetPassport | None, stats: dict) -> str:
    lines = [
        "# MindGraph++ Dataset Quality Report\n",
        "## Dataset Overview\n",
        f"- **Approved:** {result.approved}\n",
        f"- **Gates passed:** {result.passed_gates}/{len(result.gates)}\n",
    ]
    if passport:
        lines.extend(
            [
                f"- **Participants:** {passport.participants}\n",
                f"- **Sessions:** {passport.sessions}\n",
                f"- **Quality score:** {passport.quality_score}\n",
                f"- **Validation status:** {passport.validation_status}\n",
            ]
        )

    lines.append("\n## Validation Summary\n")
    for gate in result.gates:
        lines.append(f"- **{gate.gate}:** {gate.status.value} ({len(gate.issues)} issues)\n")

    lines.append("\n## Statistics\n")
    for k, v in list(stats.items())[:8]:
        lines.append(f"- {k}: {v}\n")

    lines.append("\n## Recommendations\n")
    if not result.approved:
        lines.append("- Resolve failed gates before training.\n")
    else:
        lines.append("- Dataset approved for model development.\n")

    failed = [g for g in result.gates if g.status.value == "failed"]
    if failed:
        lines.append("\n## Known Issues\n")
        for g in failed:
            for issue in g.issues[:5]:
                lines.append(f"- {g.gate}: {issue.message}\n")

    lines.append("\n## Future Improvements\n")
    lines.append("- Run full modality pipelines to populate feature store embeddings.\n")
    lines.append("- Re-validate after embedding extraction at scale.\n")
    return "".join(lines)


def write_reports(result: DataQualityResult, passport, stats: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    md = generate_master_report(result, passport, stats)
    md_path = output_dir / "dataset_quality_report.md"
    md_path.write_text(md, encoding="utf-8")
    paths["markdown"] = md_path

    html = f"<!DOCTYPE html><html><head><title>Dataset Quality</title></head><body><pre>{md}</pre></body></html>"
    html_path = output_dir / "dataset_quality_report.html"
    html_path.write_text(html, encoding="utf-8")
    paths["html"] = html_path
    return paths
