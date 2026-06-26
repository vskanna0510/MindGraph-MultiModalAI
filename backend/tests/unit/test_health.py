"""API endpoint tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.config.settings import Settings, get_settings
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


@pytest.mark.asyncio
async def test_health_endpoint(app) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["data"]["service"] == "MindGraph++"
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_root_endpoint(app) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["documentation"] == "/docs"
