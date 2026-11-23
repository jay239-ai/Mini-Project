"""Endpoints for importing and reading site-specific historic traffic reports."""

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.reporting.historical_reports import MAX_PDF_BYTES, get_site_history, save_report

router = APIRouter(prefix="/history", tags=["historical traffic context"])


@router.post("/reports", status_code=201)
async def import_historical_report(
    site_label: str = Form(..., min_length=1, max_length=120),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Extract supported numeric facts from a PDF; keep no original document/text."""

    if file.content_type not in {"application/pdf", "application/octet-stream"} and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Choose a PDF traffic report.")
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_PDF_BYTES:
                raise HTTPException(status_code=413, detail="The PDF exceeds the 20 MB upload limit.")
            chunks.append(chunk)
        return save_report(site_label, file.filename or "traffic-report.pdf", b"".join(chunks))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()


@router.get("/context")
def historical_context(site_label: str = "Junction A · North") -> dict[str, Any]:
    try:
        return get_site_history(site_label)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
