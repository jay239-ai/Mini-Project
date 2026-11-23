"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.history import router as history_router
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
API_PREFIX = "/api/v1"

app = FastAPI(
    title="Free-Left Traffic Dashboard",
    version="0.1.0",
    description="Backend foundation for configurable live and recorded traffic analysis.",
)
app.include_router(health_router, prefix=API_PREFIX)
app.include_router(analysis_router, prefix=API_PREFIX)
app.include_router(history_router, prefix=API_PREFIX)


@app.middleware("http")
async def prevent_stale_frontend_cache(request: Request, call_next):
    """Always serve current dashboard assets during local development."""

    response = await call_next(request)
    if request.url.path == "/dashboard.html" or request.url.path.startswith(("/css/", "/js/")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the integrated frontend landing page."""

    return FileResponse(FRONTEND_DIR / "index.html")


# Keep this last so API routes and the OpenAPI pages take precedence.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend-assets")
