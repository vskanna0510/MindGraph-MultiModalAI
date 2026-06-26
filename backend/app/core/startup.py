"""Startup validation — fail fast on misconfiguration."""

from __future__ import annotations

import os
from pathlib import Path

from app.config.settings import Settings
from app.logging import get_logger

logger = get_logger(__name__)

REQUIRED_CONFIG_FILES = (
    "backend.yaml",
    "database.yaml",
    "security.yaml",
    "logging.yaml",
)

REQUIRED_DIRECTORIES = (
    "logs",
    "cache",
    "cache/embeddings",
    "cache/inference",
    "cache/downloads",
    "cache/neo4j",
    "cache/features",
    "uploads",
    "exports",
    "reports",
    "backups",
    "backups/postgres",
    "backups/neo4j",
    "ml_pipeline/weights",
    "datasets",
    "embeddings",
)


class StartupValidationError(RuntimeError):
    """Raised when startup validation fails."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_directories(settings: Settings) -> list[str]:
    """Create required runtime directories if missing."""
    root = _project_root()
    created: list[str] = []
    for relative in REQUIRED_DIRECTORIES:
        path = root / relative
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(relative)
    log_dir = root / settings.log_directory.removeprefix("./")
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(log_dir.relative_to(root)))
    return created


def validate_config_files(settings: Settings) -> None:
    """Verify required YAML configuration files exist."""
    config_dir = _project_root() / settings.config_directory.removeprefix("./")
    missing = [name for name in REQUIRED_CONFIG_FILES if not (config_dir / name).exists()]
    if missing:
        raise StartupValidationError(f"Missing configuration files: {', '.join(missing)}")


def validate_environment_settings(settings: Settings) -> None:
    """Validate critical environment variables."""
    if settings.app_env == "production" and settings.app_debug:
        raise StartupValidationError("APP_DEBUG must be false in production")
    if len(settings.jwt_secret) < 32:
        raise StartupValidationError("JWT_SECRET must be at least 32 characters")


async def validate_dependencies(settings: Settings) -> dict[str, str]:
    """Check connectivity to optional infrastructure dependencies."""
    from app.database.health import check_neo4j, check_postgres, check_redis

    results: dict[str, str] = {}
    results["postgres"] = await check_postgres(settings)
    results["redis"] = await check_redis(settings)
    results["neo4j"] = await check_neo4j(settings)
    return results


async def run_startup_validation(settings: Settings) -> None:
    """Run all startup checks; raise on failure."""
    if os.getenv("SKIP_STARTUP_VALIDATION", "").lower() in {"1", "true", "yes"}:
        logger.warning("startup_validation_skipped")
        return

    validate_environment_settings(settings)
    validate_config_files(settings)
    created = ensure_directories(settings)
    if created:
        logger.info("directories_created", paths=created)

    if settings.app_env in {"development", "staging", "production"}:
        deps = await validate_dependencies(settings)
        unhealthy = [name for name, status in deps.items() if status != "healthy"]
        if unhealthy and settings.app_env == "production":
            raise StartupValidationError(
                f"Unhealthy dependencies in production: {', '.join(unhealthy)}"
            )
        if unhealthy:
            logger.warning("dependencies_not_ready", dependencies=deps)

    logger.info("startup_validation_passed", environment=settings.app_env)
