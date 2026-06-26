"""Shared API response schemas."""

from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str
    message: str
    field: str | None = None


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response envelope."""

    status: str = Field(description="success or error")
    message: str
    data: T | None = None
    errors: list[ErrorDetail] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def success(cls, message: str, data: T | None = None) -> "ApiResponse[T]":
        return cls(status="success", message=message, data=data)

    @classmethod
    def failure(
        cls,
        message: str,
        errors: list[ErrorDetail] | None = None,
    ) -> "ApiResponse[None]":
        return cls(status="error", message=message, errors=errors or [])


class HealthStatus(BaseModel):
    """Health check response data."""

    service: str
    version: str
    environment: str
    dependencies: dict[str, str] = Field(default_factory=dict)


class ServiceInfo(BaseModel):
    """Root service information."""

    name: str
    version: str
    api_version: str
    documentation: str
    health: str
