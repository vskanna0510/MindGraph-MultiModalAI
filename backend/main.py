"""Uvicorn entry point for the MindGraph++ backend.

Run from the backend directory:

    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from app.main import app

__all__ = ["app"]
