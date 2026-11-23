"""Local, text-only import and SQLite storage for historical traffic reports."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HISTORY_DB = PROJECT_ROOT / "data" / "processed" / "traffic_history.sqlite3"
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_EXTRACTED_CHARS = 2_000_000

_DURATION = re.compile(r"(?P<value>\d{1,5}(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|hr|minutes?|mins?|min|seconds?|secs?|sec)\b", re.I)
_DATE = re.compile(r"\b(?:20\d{2}-\d{1,2}-\d{1,2}|\d{1,2}[/.\-]\d{1,2}[/.\-]20\d{2})\b")


def _connect() -> sqlite3.Connection:
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS historical_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_key TEXT NOT NULL,
            site_label TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            report_period_start TEXT,
            report_period_end TEXT,
            incident_count INTEGER,
            violation_count INTEGER,
            maximum_queue_vehicles INTEGER,
            maximum_queue_length_meters REAL,
            average_delay_minutes REAL,
            longest_delay_minutes REAL,
            peak_congestion_level TEXT,
            peak_period TEXT,
            extraction_confidence TEXT NOT NULL,
            extraction_warnings TEXT NOT NULL,
            UNIQUE(site_key, content_sha256)
        )
    """)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(historical_reports)")}
    for column, column_type in {"maximum_queue_vehicles": "INTEGER", "maximum_queue_length_meters": "REAL"}.items():
        if column not in columns:
            connection.execute(f"ALTER TABLE historical_reports ADD COLUMN {column} {column_type}")
    connection.commit()
    return connection


@contextmanager
def _database() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def normalize_site_key(site_label: str) -> str:
    normalized = "".join(character.casefold() if character.isalnum() else " " for character in site_label)
    key = "_".join(normalized.split())
    if not key or len(key) > 120:
        raise ValueError("Enter a valid site or intersection name (1–120 characters).")
    return key


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes or len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF must be non-empty and no larger than 20 MB.")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF document.")
    try:
        import fitz
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError("The PDF could not be opened. It may be damaged, encrypted, or unsupported.") from error
    try:
        if document.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported. Export an unlocked copy first.")
        if document.page_count > MAX_PDF_PAGES:
            raise ValueError("The PDF is too long to process (maximum 500 pages).")
        chunks: list[str] = []
        size = 0
        for page in document:
            text = page.get_text("text", sort=True)
            size += len(text)
            if size > MAX_EXTRACTED_CHARS:
                raise ValueError("The extracted report text is too large to process safely.")
            chunks.append(text)
        extracted = "\n".join(chunks).strip()
        if len(re.sub(r"\s", "", extracted)) < 40:
            raise ValueError("No readable report text was found. This looks scanned or image-only; text-based PDFs are supported, OCR is not enabled.")
        return extracted
    finally:
        document.close()


def _duration_minutes(value: float, unit: str) -> float:
    unit = unit.casefold()
    if unit.startswith("h"):
        return value * 60
    if unit.startswith("s"):
        return value / 60
    return value


def parse_traffic_report(text: str) -> dict[str, Any]:
    """Extract only label-led facts or jam-associated durations; never store raw text."""

    clean = re.sub(r"[\t\u00a0]+", " ", text)
    lines = [re.sub(r"\s+", " ", line).strip(" :-\u2022") for line in clean.splitlines()]
    lines = [line for line in lines if line]
    joined = "\n".join(lines)

    incident_count = None
    labeled_count_patterns = (
        r"\b(?:total|number of)\s+(?:traffic\s+)?(?:jam|congestion|blockage|incident)s?\s*(?:events?|records?)?\s*[:=\-]?\s*(\d{1,7})\b",
        r"\b(\d{1,7})\s+(?:traffic\s+)?(?:jam|congestion|blockage|incident)s?\s+(?:events?|records?)\b",
    )
    for pattern in labeled_count_patterns:
        match = re.search(pattern, joined, re.I)
        if match:
            incident_count = int(match.group(1))
            break
    if incident_count is None:
        match = re.search(r"\b(?:traffic\s+)?(?:jam|congestion|blockage|incident)s?\s*(?:count|events?|records?)\s*[:=\-]\s*(\d{1,7})\b", joined, re.I)
        if match:
            incident_count = int(match.group(1))

    violation_count = None
    violation_match = re.search(r"\b(?:total\s+)?(?:traffic\s+)?violations?\s*(?:records?|count)?\s*[:=\-]\s*(\d{1,7})\b", joined, re.I)
    if violation_match:
        violation_count = int(violation_match.group(1))

    peak_level = None
    level_match = re.search(r"\b(?:peak[- ]hour\s+)?congestion(?:\s+level)?\s*[:=]\s*(low|moderate|medium|high|severe|very\s+high)\b", joined, re.I)
    if level_match:
        peak_level = re.sub(r"\s+", " ", level_match.group(1)).casefold()

    peak_period = None
    period_match = re.search(r"\bpeak(?:[- ]hour|[- ]period)?\s*(?:congestion\s+)?(?:period\s*)?[:=]\s*([^\n]{1,80})", joined, re.I)
    if period_match and period_match.group(1).strip(" .;,").casefold() not in {"low", "moderate", "medium", "high", "severe", "very high"}:
        peak_period = period_match.group(1).strip(" .;,")

    maximum_queue_vehicles = None
    queue_match = re.search(r"\b(?:maximum|max|peak|largest)\s+queue(?:\s+(?:length|size))?\s*[:=]\s*(\d{1,6})\s*(?:vehicles?|cars?)?\b", joined, re.I)
    if queue_match:
        maximum_queue_vehicles = int(queue_match.group(1))
    maximum_queue_length_meters = None
    length_match = re.search(r"\b(?:(?:maximum|max|peak|longest)\s+)?(?:queue\s+length|length\s+of\s+(?:the\s+)?queue)\s*[:=]\s*(\d{1,7}(?:\.\d+)?)\s*(?:m|meters?|metres?)\b", joined, re.I)
    if length_match:
        maximum_queue_length_meters = float(length_match.group(1))

    report_dates: list[str] = []
    for raw_date in _DATE.findall(joined):
        try:
            parsed = datetime.strptime(raw_date, "%Y-%m-%d") if "-" in raw_date and raw_date[:4].isdigit() else None
            if parsed is None:
                parsed = next(datetime.strptime(raw_date, fmt) for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y") if _matches_format(raw_date, fmt))
            report_dates.append(parsed.date().isoformat())
        except (ValueError, StopIteration):
            continue
    report_dates = sorted(set(report_dates))

    durations: list[float] = []
    event_lines = [line for line in lines if re.search(r"\b(jam|congest|traffic\s+delay|blockage|blocked|incident)\b", line, re.I)]
    for line in event_lines:
        for match in _DURATION.finditer(line):
            durations.append(_duration_minutes(float(match.group("value")), match.group("unit")))

    # Labeled aggregate duration fields are useful even when a report is not an event log.
    labeled_durations: dict[str, float] = {}
    for label, pattern in {
        "average": r"\b(?:average|mean)\s+(?:(?:traffic\s+)?(?:jam|congestion)\s+)?(?:delay|duration|time|wait|waiting)\s*[:=]\s*(\d{1,5}(?:\.\d+)?)\s*(hours?|hrs?|hr|minutes?|mins?|min|seconds?|secs?|sec)\b",
        "longest": r"\b(?:longest|maximum|max)\s+(?:(?:traffic\s+)?(?:jam|congestion)\s+)?(?:delay|duration|time|wait|waiting)\s*[:=]\s*(\d{1,5}(?:\.\d+)?)\s*(hours?|hrs?|hr|minutes?|mins?|min|seconds?|secs?|sec)\b",
    }.items():
        match = re.search(pattern, joined, re.I)
        if match:
            labeled_durations[label] = _duration_minutes(float(match.group(1)), match.group(2))

    evidence_count = sum(value is not None for value in (incident_count, violation_count, peak_level, maximum_queue_vehicles, maximum_queue_length_meters)) + bool(durations) + bool(labeled_durations)
    if evidence_count == 0:
        raise ValueError("The PDF contains text, but no supported traffic fields were recognized. Use a report with labeled congestion/jam counts, durations, violation counts, or congestion level.")
    warnings: list[str] = []
    if not report_dates:
        warnings.append("No report dates were recognized; history will be retained without a recency claim.")
    if not durations and not labeled_durations:
        warnings.append("No congestion duration was recognized.")
    if peak_level is None:
        warnings.append("No labeled congestion level was recognized.")
    average_delay = labeled_durations.get("average", round(sum(durations) / len(durations), 2) if durations else None)
    longest_delay = labeled_durations.get("longest", max(durations, default=None))
    confidence = "high" if evidence_count >= 3 and report_dates else "moderate" if evidence_count >= 2 else "low"
    return {
        "incident_count": incident_count,
        "violation_count": violation_count,
        "maximum_queue_vehicles": maximum_queue_vehicles,
        "maximum_queue_length_meters": maximum_queue_length_meters,
        "average_delay_minutes": average_delay,
        "longest_delay_minutes": longest_delay,
        "peak_congestion_level": peak_level,
        "peak_period": peak_period,
        "report_period_start": report_dates[0] if report_dates else None,
        "report_period_end": report_dates[-1] if report_dates else None,
        "extraction_confidence": confidence,
        "extraction_warnings": warnings,
    }


def _matches_format(value: str, fmt: str) -> bool:
    try:
        datetime.strptime(value, fmt)
        return True
    except ValueError:
        return False


def save_report(site_label: str, filename: str, pdf_bytes: bytes) -> dict[str, Any]:
    site_key = normalize_site_key(site_label)
    extracted = extract_pdf_text(pdf_bytes)
    facts = parse_traffic_report(extracted)
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    # Do not persist user-supplied filenames; these may contain personal details.
    safe_name = "Traffic report"
    imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    values = (
        site_key, site_label.strip()[:120], safe_name, digest, imported_at,
        facts["report_period_start"], facts["report_period_end"], facts["incident_count"],
        facts["violation_count"], facts["maximum_queue_vehicles"], facts["maximum_queue_length_meters"],
        facts["average_delay_minutes"], facts["longest_delay_minutes"],
        facts["peak_congestion_level"], facts["peak_period"], facts["extraction_confidence"],
        json.dumps(facts["extraction_warnings"]),
    )
    with _database() as connection:
        try:
            cursor = connection.execute("""
                INSERT INTO historical_reports (
                    site_key, site_label, filename, content_sha256, imported_at,
                    report_period_start, report_period_end, incident_count, violation_count,
                    maximum_queue_vehicles, maximum_queue_length_meters,
                    average_delay_minutes, longest_delay_minutes, peak_congestion_level,
                    peak_period, extraction_confidence, extraction_warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)
        except sqlite3.IntegrityError:
            row = connection.execute("SELECT id FROM historical_reports WHERE site_key=? AND content_sha256=?", (site_key, digest)).fetchone()
            return {"status": "duplicate", "report_id": row["id"], **facts, "site_label": site_label.strip()}
        report_id = cursor.lastrowid
    return {"status": "imported", "report_id": report_id, "site_label": site_label.strip(), **facts}


def get_site_history(site_label: str) -> dict[str, Any]:
    site_key = normalize_site_key(site_label)
    with _database() as connection:
        rows = connection.execute("SELECT * FROM historical_reports WHERE site_key=? ORDER BY imported_at DESC, id DESC", (site_key,)).fetchall()
    if not rows:
        return {"site_label": site_label, "matched_site": False, "report_count": 0, "incident_count": 0, "violation_count": 0, "peak_congestion_level": None, "average_delay_minutes": None, "longest_delay_minutes": None, "peak_period": None, "confidence": "none"}
    level_rank = {"low": 0, "moderate": 1, "medium": 1, "high": 2, "very high": 3, "severe": 3}
    levels = [str(row["peak_congestion_level"]).casefold() for row in rows if row["peak_congestion_level"]]
    peak_level = max(levels, key=lambda level: level_rank.get(level, 0), default=None)
    confidences = [row["extraction_confidence"] for row in rows]
    latest = rows[0]
    cutoff = date.today() - timedelta(days=365)
    context_rows = []
    for row in rows:
        try:
            report_date = date.fromisoformat(str(row["report_period_end"] or ""))
            if cutoff <= report_date <= date.today() and row["extraction_confidence"] in {"moderate", "high"}:
                context_rows.append(row)
        except ValueError:
            continue
    context_levels = [str(row["peak_congestion_level"]).casefold() for row in context_rows if row["peak_congestion_level"]]
    context_peak_level = max(context_levels, key=lambda level: level_rank.get(level, 0), default=None)
    decision_context = {
        "matched_site": bool(context_rows),
        "report_count": len(context_rows),
        "confidence": "high" if any(row["extraction_confidence"] == "high" for row in context_rows) else "moderate" if context_rows else "none",
        "peak_congestion_level": context_peak_level,
        "maximum_queue_vehicles": max((int(row["maximum_queue_vehicles"]) for row in context_rows if row["maximum_queue_vehicles"] is not None), default=None),
        "report_period_end": max((row["report_period_end"] for row in context_rows), default=None),
    }
    return {
        "site_label": rows[0]["site_label"],
        "latest_filename": "Traffic report",
        "matched_site": True,
        "report_count": len(rows),
        # Counts from overlapping or differently scoped reports are not summed.
        "incident_count": int(latest["incident_count"]) if latest["incident_count"] is not None else None,
        "violation_count": int(latest["violation_count"]) if latest["violation_count"] is not None else None,
        "maximum_queue_vehicles": max((int(row["maximum_queue_vehicles"]) for row in rows if row["maximum_queue_vehicles"] is not None), default=None),
        "maximum_queue_length_meters": max((float(row["maximum_queue_length_meters"]) for row in rows if row["maximum_queue_length_meters"] is not None), default=None),
        "average_delay_minutes": float(latest["average_delay_minutes"]) if latest["average_delay_minutes"] is not None else None,
        "longest_delay_minutes": float(latest["longest_delay_minutes"]) if latest["longest_delay_minutes"] is not None else None,
        "peak_congestion_level": peak_level,
        "peak_period": next((row["peak_period"] for row in rows if row["peak_period"]), None),
        "report_period_start": min((row["report_period_start"] for row in rows if row["report_period_start"]), default=None),
        "report_period_end": max((row["report_period_end"] for row in rows if row["report_period_end"]), default=None),
        "latest_imported_at": rows[0]["imported_at"],
        "decision_context": decision_context,
        "count_basis": "latest report only; reports are not summed because reporting periods may overlap",
        "reports": [{
            "report_period_start": row["report_period_start"],
            "report_period_end": row["report_period_end"],
            "incident_count": row["incident_count"],
            "violation_count": row["violation_count"],
            "maximum_queue_vehicles": row["maximum_queue_vehicles"],
            "maximum_queue_length_meters": row["maximum_queue_length_meters"],
            "average_delay_minutes": row["average_delay_minutes"],
            "longest_delay_minutes": row["longest_delay_minutes"],
            "peak_congestion_level": row["peak_congestion_level"],
            "extraction_confidence": row["extraction_confidence"],
        } for row in rows],
        "confidence": "high" if "high" in confidences else "moderate" if "moderate" in confidences else "low",
        "warnings": sorted({warning for row in rows for warning in json.loads(row["extraction_warnings"])}),
        "raw_pdf_retained": False,
        "raw_text_retained": False,
    }
