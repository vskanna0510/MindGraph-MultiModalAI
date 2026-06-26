"""Health check and service information endpoints."""

from fastapi import APIRouter, Depends

from app import __version__
from app.config.settings import Settings, get_settings
from app.schemas.common import ApiResponse, HealthStatus, ServiceInfo

router = APIRouter(tags=["health"])


@router.get("/", response_model=ApiResponse[ServiceInfo])
async def root(settings: Settings = Depends(get_settings)) -> ApiResponse[ServiceInfo]:
    """Return service metadata and navigation links."""
    info = ServiceInfo(
        name=settings.app_name,
        version=__version__,
        api_version=settings.api_version,
        documentation="/docs",
        health=f"/api/{settings.api_version}/health",
    )
    return ApiResponse.success(message="MindGraph++ API", data=info)


@router.get("/health", response_model=ApiResponse[HealthStatus])
async def health_check(settings: Settings = Depends(get_settings)) -> ApiResponse[HealthStatus]:
    """Return application health status."""
    app_config = settings.get_yaml("app.yaml")
    include_deps = app_config.get("health", {}).get("include_dependencies", True)

    dependencies: dict[str, str] = {}
    if include_deps:
        dependencies = {
            "postgres": "not_checked",
            "redis": "not_checked",
            "neo4j": "not_checked",
        }

    status = HealthStatus(
        service=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        dependencies=dependencies,
    )
    return ApiResponse.success(message="Service is healthy", data=status)
