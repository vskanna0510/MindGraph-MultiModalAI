"""Global exception handlers for the FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import MindGraphError
from app.logging import get_logger
from app.schemas.common import ApiResponse, ErrorDetail

logger = get_logger(__name__)


def register_exception_handlers(application: FastAPI) -> None:
    """Register handlers that return the standard API envelope.

    Parameters:
        application: FastAPI app instance.

    Side Effects:
        Mutates application exception handler registry.
    """

    @application.exception_handler(MindGraphError)
    async def mindgraph_error_handler(
        request: Request,
        exc: MindGraphError,
    ) -> JSONResponse:
        logger.warning(
            "application_error",
            code=exc.code,
            message=exc.message,
            path=request.url.path,
        )
        body = ApiResponse.failure(
            message=exc.message,
            errors=[ErrorDetail(code=exc.code, message=exc.message)],
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = [
            ErrorDetail(
                code="validation_error",
                message=str(error.get("msg", "Invalid input")),
                field=".".join(str(part) for part in error.get("loc", [])[1:]) or None,
            )
            for error in exc.errors()
        ]
        logger.warning("request_validation_failed", path=request.url.path, error_count=len(errors))
        body = ApiResponse.failure(message="Request validation failed", errors=errors)
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", path=request.url.path)
        body = ApiResponse.failure(
            message="An unexpected error occurred. Please try again later.",
            errors=[ErrorDetail(code="internal_error", message="Internal server error")],
        )
        return JSONResponse(status_code=500, content=body.model_dump(mode="json"))
