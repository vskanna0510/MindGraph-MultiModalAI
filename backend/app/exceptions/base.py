"""Custom exception hierarchy for MindGraph++.

Purpose:
    Provide typed, loggable exceptions mapped to HTTP responses without
    exposing stack traces to clients.

Responsibilities:
    - Encode error codes for API consumers
    - Carry safe user-facing messages
    - Support structured logging in middleware
"""

from __future__ import annotations


class MindGraphError(Exception):
    """Base exception for all MindGraph++ application errors.

    Parameters:
        message: Safe user-facing error description.
        code: Machine-readable error code.
        status_code: HTTP status to return.
    """

    def __init__(
        self,
        message: str,
        code: str = "internal_error",
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class AuthenticationError(MindGraphError):
    """Raised when authentication fails or tokens are invalid."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message=message, code="authentication_error", status_code=401)


class AuthorizationError(MindGraphError):
    """Raised when the user lacks permission for an action."""

    def __init__(self, message: str = "Access denied") -> None:
        super().__init__(message=message, code="authorization_error", status_code=403)


class ValidationError(MindGraphError):
    """Raised when request or domain validation fails."""

    def __init__(self, message: str = "Validation failed") -> None:
        super().__init__(message=message, code="validation_error", status_code=422)


class InferenceError(MindGraphError):
    """Raised when the ML inference pipeline fails."""

    def __init__(self, message: str = "Inference could not be completed") -> None:
        super().__init__(message=message, code="inference_error", status_code=503)


class GraphError(MindGraphError):
    """Raised when Neo4j knowledge graph operations fail."""

    def __init__(self, message: str = "Knowledge graph operation failed") -> None:
        super().__init__(message=message, code="graph_error", status_code=503)


class DatasetError(MindGraphError):
    """Raised when dataset loading or preprocessing fails."""

    def __init__(self, message: str = "Dataset operation failed") -> None:
        super().__init__(message=message, code="dataset_error", status_code=400)


class StorageError(MindGraphError):
    """Raised when object or file storage operations fail."""

    def __init__(self, message: str = "Storage operation failed") -> None:
        super().__init__(message=message, code="storage_error", status_code=503)


class SecurityError(MindGraphError):
    """Raised when a security policy violation is detected."""

    def __init__(self, message: str = "Security policy violation") -> None:
        super().__init__(message=message, code="security_error", status_code=403)
