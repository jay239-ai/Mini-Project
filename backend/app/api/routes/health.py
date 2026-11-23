"""Liveness and readiness endpoints."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Check whether the API process is running")
def liveness() -> dict[str, str]:
    """Liveness does not depend on PostgreSQL or camera availability."""

    return {"status": "ok"}
