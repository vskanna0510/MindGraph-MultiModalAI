"""Data quality domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class ValidationIssue:
    check: str
    severity: str
    message: str
    participant_id: str = ""
    session_id: str = ""


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScore:
    audio: float = 0.0
    video: float = 0.0
    transcript: float = 0.0
    synchronization: float = 0.0
    embedding: float = 0.0
    metadata: float = 0.0
    overall: float = 0.0
    category: str = "acceptable"


@dataclass
class AuditEntry:
    validation_start: str
    validation_end: str
    user: str
    machine: str
    configuration_hash: str
    pipeline_version: str
    warnings: int
    errors: int
    approval_status: ApprovalStatus
    gates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetPassport:
    dataset_name: str
    version: str
    hash: str
    creation_date: str
    source: str
    license: str
    num_samples: int
    participants: int
    sessions: int
    languages: list[str]
    modalities: list[str]
    validation_status: str
    quality_score: float
    processing_version: str
    git_commit: str
    random_seed: int
    feature_version: str = "v1.0"
    embedding_version: str = "v1.0"


@dataclass
class DataQualityResult:
    gates: list[GateResult] = field(default_factory=list)
    passport: DatasetPassport | None = None
    audit: AuditEntry | None = None
    approved: bool = False
    report_paths: dict[str, str] = field(default_factory=dict)

    @property
    def passed_gates(self) -> int:
        return sum(1 for g in self.gates if g.status == GateStatus.PASSED)

    def to_ci_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "passed_gates": self.passed_gates,
            "total_gates": len(self.gates),
            "gates": [
                {"gate": g.gate, "status": g.status.value, "issues": len(g.issues)}
                for g in self.gates
            ],
        }
