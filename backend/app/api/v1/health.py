"""Health check and service information endpoints."""

from fastapi import APIRouter, Depends, Response, status

from app import __version__
from app.config.settings import Settings, get_settings
from app.database.health import check_neo4j, check_postgres, check_redis
from app.schemas.common import ApiResponse, HealthStatus, ServiceInfo
from app.schemas.health import LivenessStatus, ReadinessStatus

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
    """Return aggregate application health status."""
    backend_config = settings.get_yaml("backend.yaml")
    app_config = settings.get_yaml("app.yaml")
    include_deps = backend_config.get("health", {}).get("include_dependencies")
    if include_deps is None:
        include_deps = app_config.get("health", {}).get("include_dependencies", True)

    dependencies: dict[str, str] = {}
    if include_deps:
        dependencies = {
            "postgres": await check_postgres(settings),
            "redis": await check_redis(settings),
            "neo4j": await check_neo4j(settings),
        }

    payload = HealthStatus(
        service=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        dependencies=dependencies,
    )
    return ApiResponse.success(message="Service is healthy", data=payload)


@router.get("/health/liveness", response_model=ApiResponse[LivenessStatus])
async def liveness(settings: Settings = Depends(get_settings)) -> ApiResponse[LivenessStatus]:
    """Kubernetes liveness probe — process is running."""
    data = LivenessStatus(alive=True, service=settings.app_name, version=__version__)
    return ApiResponse.success(message="Alive", data=data)


@router.get("/health/readiness", response_model=ApiResponse[ReadinessStatus])
async def readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> ApiResponse[ReadinessStatus]:
    """Kubernetes readiness probe — dependencies available."""
    dependencies = {
        "postgres": await check_postgres(settings),
        "redis": await check_redis(settings),
        "neo4j": await check_neo4j(settings),
    }
    ready = all(state == "healthy" for state in dependencies.values())
    data = ReadinessStatus(ready=ready, dependencies=dependencies)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ApiResponse.success(
        message="Ready" if ready else "Not ready",
        data=data,
    )
