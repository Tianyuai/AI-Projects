"""Static browser UI mounted into the canonical FastAPI application."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


_STATIC_DIRECTORY = Path(__file__).parent / "static"


def install_ui(app: FastAPI) -> None:
    """Install the browser UI without adding a second search composition path."""
    app.mount("/static", StaticFiles(directory=_STATIC_DIRECTORY), name="static")

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(_STATIC_DIRECTORY / "index.html")
