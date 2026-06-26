"""Tests for custom exceptions and error handlers."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.config.settings import Settings, get_settings
from app.exceptions import AuthenticationError, ValidationError
from app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        secret_key="test-secret-key-min-16-chars",
        encryption_key="test-encryption-key-32bytes!!",
        jwt_secret="test-jwt-secret-minimum-32-characters-long",
        database_url="postgresql+asyncpg://mindgraph:mindgraph@localhost:5432/mindgraph_test",
        app_env="testing",
        log_format="console",
    )


@pytest.fixture
def app(test_settings: Settings):
    get_settings.cache_clear()
    application = create_app(test_settings)
    application.dependency_overrides[get_settings] = lambda: test_settings
    yield application
    application.dependency_overrides.clear()
    get_settings.cache_clear()


def test_exception_attributes() -> None:
    error = AuthenticationError("Invalid token")
    assert error.status_code == 401
    assert error.code == "authentication_error"
    assert error.message == "Invalid token"


def test_validation_exception_defaults() -> None:
    error = ValidationError()
    assert error.status_code == 422


@pytest.mark.asyncio
async def test_invalid_route_returns_envelope(app) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
