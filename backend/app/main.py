"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1 import api_v1_router
from app.config.settings import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.middleware.request_context import RequestContextMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    configure_logging(settings)
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
    )
    yield
    logger.info("application_shutdown", app_name=settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or get_settings()
    app_config = resolved_settings.get_yaml("app.yaml")

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description=(
            "Privacy-preserving multilingual multimodal longitudinal "
            "depression intelligence API. Not a medical diagnosis service."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origin_list,
        allow_credentials=app_config.get("cors", {}).get("allow_credentials", True),
        allow_methods=app_config.get("cors", {}).get("allow_methods", ["*"]),
        allow_headers=app_config.get("cors", {}).get("allow_headers", ["*"]),
    )
    application.add_middleware(RequestContextMiddleware)

    application.include_router(
        api_v1_router,
        prefix=f"/api/{resolved_settings.api_version}",
    )
    application.state.settings = resolved_settings

    return application


app = create_app()
