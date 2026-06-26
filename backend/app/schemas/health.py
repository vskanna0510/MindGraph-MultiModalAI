"""Extended health check schemas."""

from pydantic import BaseModel, Field


class DependencyHealth(BaseModel):
    """Health status of an infrastructure dependency."""

    name: str
    status: str


class ReadinessStatus(BaseModel):
    """Readiness probe response data."""

    ready: bool
    dependencies: dict[str, str] = Field(default_factory=dict)


class LivenessStatus(BaseModel):
    """Liveness probe response data."""

    alive: bool
    service: str
    version: str
