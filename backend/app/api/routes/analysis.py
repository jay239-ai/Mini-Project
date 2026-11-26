"""Recorded three-camera analysis jobs for the dashboard."""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.app.decision_engine.rules import DecisionThresholds, recommend, valid_zone_configuration
from backend.app.reporting.session_pdf import generate_session_pdf
from backend.app.reporting.historical_reports import get_site_history
from vision_engine.camera.video_analysis import TARGET_FPS, analyze_video, open_video_capture, _draw_performance_hud
from vision_engine.camera.live_capture import enumerate_usb_cameras, run_live_camera
from vision_engine.detection.yolo import TRAFFIC_CLASS_IDS, UltralyticsByteTrackDetector
from vision_engine.roi.zones import Zone, zones_from_mapping
from vision_engine.metrics.cross_camera import associate_camera_tracks, LiveCrossCameraAssociator

PROJECT_ROOT = Path(__file__).resolve().parents[4]
JOB_ROOT = PROJECT_ROOT / "data" / "processed" / "dashboard_jobs"
ZONE_CONFIG = PROJECT_ROOT / "config" / "camera_zones.example.json"
CAMERAS = ("approach", "turning", "exit")
JOBS: dict[str, dict[str, Any]] = {}
RUNNING_TASKS: set[asyncio.Task[None]] = set()
LIVE_SESSION_LOCK = threading.Lock()
router = APIRouter(prefix="/analysis", tags=["recorded analysis"])
LOGGER = logging.getLogger(__name__)
DEFAULT_SITE_LABEL = "Junction A · North"


def _site_history(site_label: str) -> dict[str, Any]:
    try:
        return get_site_history(site_label)
    except Exception:
        LOGGER.exception("Could not read historical traffic context for %s", site_label)
        return {"site_label": site_label, "matched_site": False, "report_count": 0, "peak_congestion_level": None, "confidence": "none"}


def _publish_job_decision(job: dict[str, Any], decision: dict[str, Any], now: float | None = None) -> None:
    """Save the current advisory and compact timing for signal color changes."""

    timestamp = time.monotonic() if now is None else now
    elapsed = round(max(0.0, timestamp - float(job.get("_decision_started_monotonic", timestamp))), 3)
    timing_defaults = {
        "update_count": 0,
        "signal_change_count": 0,
        "first_update_after_seconds": None,
        "last_update_after_seconds": None,
        "mean_update_interval_seconds": None,
        "minimum_update_interval_seconds": None,
        "maximum_update_interval_seconds": None,
        "event_history_truncated": False,
    }
    timing = job.get("decision_timing")
    if not isinstance(timing, dict):
        timing = {}
    timing = {**timing_defaults, **timing}
    job["decision_timing"] = timing
    previous_update = job.get("_decision_last_update_monotonic")
    if previous_update is not None:
        interval = max(0.0, timestamp - float(previous_update))
        total = float(job.get("_decision_interval_total_seconds", 0.0)) + interval
        interval_count = int(job.get("_decision_interval_count", 0)) + 1
        job["_decision_interval_total_seconds"] = total
        job["_decision_interval_count"] = interval_count
        for field, value in (
            ("minimum_update_interval_seconds", interval if timing["minimum_update_interval_seconds"] is None else min(float(timing["minimum_update_interval_seconds"]), interval)),
            ("maximum_update_interval_seconds", interval if timing["maximum_update_interval_seconds"] is None else max(float(timing["maximum_update_interval_seconds"]), interval)),
        ):
            timing[field] = round(value, 3)
        timing["mean_update_interval_seconds"] = round(total / interval_count, 3)
    timing["update_count"] = int(timing["update_count"]) + 1
    if timing["first_update_after_seconds"] is None:
        timing["first_update_after_seconds"] = elapsed
    timing["last_update_after_seconds"] = elapsed
    job["_decision_last_update_monotonic"] = timestamp

    light = decision.get("traffic_light") if isinstance(decision.get("traffic_light"), dict) else {}
    color = str(light.get("color") or "yellow").lower()
    previous_color = job.get("_decision_last_color")
    timeline = job.setdefault("decision_timeline", [])
    if previous_color is None or color != previous_color:
        if previous_color is not None:
            timing["signal_change_count"] = int(timing["signal_change_count"]) + 1
        if len(timeline) < 2000:
            timeline.append({
                "elapsed_seconds": elapsed,
                "color": color,
                "label": str(light.get("label") or color.upper()),
                "reason": str(decision.get("reason") or "No explanation recorded."),
                "triggered_rule": str(decision.get("triggered_rule") or "not_recorded"),
            })
        else:
            timing["event_history_truncated"] = True
        job["_decision_last_color"] = color
    job["decision"] = decision


def _decision_thresholds(config: dict[str, Any]) -> DecisionThresholds:
    values = config.get("thresholds", {})
    return DecisionThresholds(
        queue_medium=int(values.get("queue_medium_vehicles", 5)),
        queue_high=int(values.get("queue_high_vehicles", 10)),
        blockage_medium_seconds=float(values.get("blockage_medium_seconds", 10)),
        blockage_high_seconds=float(values.get("blockage_high_seconds", 30)),
        stopped_delay_medium_vehicle_seconds=float(values.get("stopped_delay_medium_vehicle_seconds", 30)),
        stopped_delay_high_vehicle_seconds=float(values.get("stopped_delay_high_vehicle_seconds", 60)),
        score_medium=int(values.get("risk_score_medium", 35)),
        score_high=int(values.get("risk_score_high", 65)),
        joining_flow_stationary_ratio_medium=float(values.get("joining_flow_stationary_ratio_medium", 0.65)),
        queue_length_medium_ratio=float(values.get("queue_length_medium_ratio", 0.35)),
        queue_length_high_ratio=float(values.get("queue_length_high_ratio", 0.65)),
        minimum_observation_seconds=float(values.get("minimum_observation_seconds", 2.0)),
        minimum_confidence_frames_per_camera=int(values.get("minimum_confidence_frames_per_camera", 30)),
        pedestrian_clearance_hold_seconds=float(values.get("pedestrian_clearance_hold_seconds", 2.0)),
    )


def _validate_zone_config(config: dict[str, Any]) -> None:
    """Fail early on malformed ROI/threshold data instead of crashing in a camera worker."""
    if not isinstance(config, dict):
        raise ValueError("Camera zone configuration must be an object.")
    resolution = config.get("reference_resolution")
    if (not isinstance(resolution, list) or len(resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 1 for value in resolution)):
        raise ValueError("Camera zone reference_resolution must contain two positive integer dimensions.")
    cameras = config.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("Camera zone configuration is missing the cameras object.")
    for camera_id in CAMERAS:
        rows = cameras.get(camera_id)
        if not isinstance(rows, list) or not valid_zone_configuration(camera_id, rows):
            raise ValueError(f"Camera zone configuration for {camera_id} is missing required zones.")
        names: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Camera zone configuration for {camera_id} contains an invalid zone.")
            name, polygon = row.get("name"), row.get("polygon")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"Camera zone names for {camera_id} must be non-empty and unique.")
            names.add(name)
            allowed_kinds = {
                "approach": {"approach", "free_left"},
                "turning": {"turn_path"},
                "exit": {"exit"},
            }[camera_id]
            if row.get("kind") not in allowed_kinds:
                raise ValueError(f"Camera zone '{name}' has an unsupported kind for {camera_id}.")
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(f"Camera zone '{name}' needs at least three points.")
            points: list[tuple[int, int]] = []
            for point in polygon:
                if (not isinstance(point, list) or len(point) != 2
                        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)):
                    raise ValueError(f"Camera zone '{name}' contains an invalid point.")
                x, y = point
                if not (0 <= x < resolution[0] and 0 <= y < resolution[1]):
                    raise ValueError(f"Camera zone '{name}' contains a point outside the reference image.")
                points.append((int(x), int(y)))
            twice_area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))))
            if twice_area == 0:
                raise ValueError(f"Camera zone '{name}' has zero polygon area.")
        if camera_id == "approach":
            lane = next(row for row in rows if row.get("kind") == "approach")
            free_left = next(row for row in rows if row.get("kind") == "free_left")
            lane_zone = Zone(
                str(lane["name"]),
                tuple((int(point[0]), int(point[1])) for point in lane["polygon"]),
                "approach",
            )
            if not all(lane_zone.contains((int(point[0]), int(point[1]))) for point in free_left["polygon"]):
                raise ValueError("Camera 1 free_left polygon must sit inside the main approach-lane polygon.")
    _decision_thresholds(config)


def _zones_for_analysis(camera_id: str, zone_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select calibrated analysis polygons for each camera role."""

    if camera_id == "approach":
        return [zone for zone in zone_rows if zone.get("kind") in {"approach", "free_left"}]
    if camera_id == "turning":
        return [zone for zone in zone_rows if zone.get("kind") == "turn_path"]
    return []


@router.post("/recorded", status_code=202)
async def start_recorded_analysis(
    camera1: UploadFile = File(...),
    camera2: UploadFile = File(...),
    camera3: UploadFile = File(...),
    performance_profile: str = Form("balanced"),
    site_label: str = Form(DEFAULT_SITE_LABEL, min_length=1, max_length=120),
) -> dict[str, str]:
    """Accept three recorded clips and process them outside the request thread."""

    if performance_profile not in {"fast", "balanced"}:
        raise HTTPException(status_code=400, detail="Choose the fast or balanced processing mode.")
    site_label = site_label.strip()
    if not site_label:
        raise HTTPException(status_code=400, detail="Enter the intersection/site name.")
    uploads = (camera1, camera2, camera3)
    for upload in uploads:
        if not upload.filename or Path(upload.filename).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            raise HTTPException(status_code=400, detail="Upload three supported video files (MP4, MOV, AVI, MKV, or WebM).")

    job_id = uuid.uuid4().hex
    input_dir = Path(tempfile.mkdtemp(prefix=f"freeleft_{job_id}_"))
    try:
        async def persist_upload(index: int, upload: UploadFile) -> Path:
            path = input_dir / f"camera{index}{Path(upload.filename or '').suffix.lower()}"
            with path.open("wb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    destination.write(chunk)
            return path

        paths = list(await asyncio.gather(*(persist_upload(index, upload) for index, upload in enumerate(uploads, start=1))))
    except Exception:
        shutil.rmtree(input_dir, ignore_errors=True)
        raise
    finally:
        for upload in uploads:
            await upload.close()

    stop_event = threading.Event()
    JOBS[job_id] = {
        "job_id": job_id,
        "source_mode": "recorded_video",
        "status": "queued",
        "message": "Uploads saved; preparing analysis models",
        "progress_percent": 0,
        "camera_progress": {camera: 0 for camera in CAMERAS},
        "live_counts_by_camera": {camera: {"vehicles_in_frame": 0, "pedestrians_in_frame": 0, "vehicle_tracks": 0, "pedestrian_tracks": 0} for camera in CAMERAS},
        "frame_sequences": {camera: 0 for camera in CAMERAS},
        "camera_model_load_seconds": {},
        "preview_jpegs": {},
        "provisional_camera_metrics": {},
        "decision": {"severity": "insufficient", "reason": "Waiting for current camera evidence."},
        "decision_timing": {},
        "decision_timeline": [{"elapsed_seconds": 0.0, "color": "yellow", "label": "YELLOW - COLLECTING EVIDENCE", "reason": "Initial cautious state while camera evidence is collected.", "triggered_rule": "initial_wait_state"}],
        "_decision_last_color": "yellow",
        "_decision_started_monotonic": time.monotonic(),
        "_stop_event": stop_event,
        "_update_lock": threading.Lock(),
        "_last_decision_update_monotonic": 0.0,
    }
    JOBS[job_id]["performance_profile"] = performance_profile
    task = asyncio.create_task(asyncio.to_thread(_run_job, job_id, input_dir, paths, performance_profile, site_label, stop_event))
    RUNNING_TASKS.add(task)
    task.add_done_callback(RUNNING_TASKS.discard)
    return {"job_id": job_id, "status": "queued"}


@router.get("/capabilities", tags=["recorded analysis"])
def analysis_capabilities() -> dict[str, Any]:
    """Report missing inference packages before the browser uploads large clips."""

    missing = [name for name in ("torch", "ultralytics") if importlib.util.find_spec(name) is None]
    cuda_available = None
    if "torch" not in missing:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            missing.append("working torch runtime")
    return {
        "recorded_analysis_ready": not missing,
        "missing_dependencies": missing,
        "compute_device": "GPU" if cuda_available else "CPU" if not missing else None,
        "live_camera_ready": not missing,
        "input_modes": ["uploaded_video", "local_usb_camera", "rtsp_camera", "http_camera"],
        "note": "Live sources can use locally attached USB webcams or RTSP/RTSPS/HTTP/HTTPS streams readable by OpenCV.",
    }


@router.get("/config/turn-path", tags=["camera calibration"])
def get_turn_path_calibration() -> dict[str, Any]:
    """Return the locally configured camera 2 path polygon for the dashboard editor."""

    try:
        config = json.loads(ZONE_CONFIG.read_text(encoding="utf-8"))
        rows = config["cameras"]["turning"]
        polygon = next((row.get("polygon", []) for row in rows if row.get("kind") == "turn_path"), [])
        return {
            "reference_resolution": config["reference_resolution"],
            "polygon": polygon,
            "configured": len(polygon) >= 3,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOGGER.exception("Could not read camera 2 turn-path calibration")
        raise HTTPException(status_code=500, detail="Could not read the local camera zone configuration.") from error


@router.put("/config/turn-path", tags=["camera calibration"])
def save_turn_path_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist camera 2's polygon in reference-frame pixels."""

    polygon = payload.get("polygon")
    if not isinstance(polygon, list) or (polygon and len(polygon) < 3):
        raise HTTPException(status_code=422, detail="Draw at least three points, or clear the polygon completely.")
    try:
        config = json.loads(ZONE_CONFIG.read_text(encoding="utf-8"))
        config["cameras"]["turning"] = [] if not polygon else [{
            "name": "free_left_turn_path",
            "kind": "turn_path",
            "polygon": polygon,
        }]
        _validate_zone_config(config)
        temporary_path = ZONE_CONFIG.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(ZONE_CONFIG)
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "configured": bool(polygon),
        "polygon": polygon,
        "reference_resolution": config["reference_resolution"],
        "message": "Camera 2 turn path saved." if polygon else "Camera 2 turn path cleared; the signal recommendation will remain cautious until calibrated.",
    }


@router.get("/devices", tags=["live analysis"])
def list_local_cameras() -> dict[str, Any]:
    """Find attached cameras that Windows exposes through OpenCV's DirectShow API."""

    try:
        return {"devices": enumerate_usb_cameras()}
    except Exception as error:
        LOGGER.exception("Could not enumerate local camera devices")
        raise HTTPException(status_code=503, detail=f"Could not enumerate USB cameras: {error}") from error


@router.post("/live/start", status_code=202, tags=["live analysis"])
async def start_live_analysis(
    camera1: str | None = Form(None),
    camera2: str | None = Form(None),
    camera3: str | None = Form(None),
    performance_profile: str = Form("balanced"),
    site_label: str = Form(DEFAULT_SITE_LABEL, min_length=1, max_length=120),
) -> dict[str, Any]:
    """Start continuous local camera processing with latest-frame buffering."""

    if performance_profile not in {"fast", "balanced"}:
        raise HTTPException(status_code=400, detail="Choose the fast or balanced processing mode.")
    site_label = site_label.strip()
    if not site_label:
        raise HTTPException(status_code=400, detail="Enter the intersection/site name.")
    device_map: dict[str, int | str] = {}
    for camera, raw_source in zip(CAMERAS, (camera1, camera2, camera3)):
        if raw_source is None or not raw_source.strip():
            continue
        source = raw_source.strip()
        if source.isdecimal():
            device_index = int(source)
            if device_index >= 8:
                raise HTTPException(status_code=400, detail="Choose a camera from the connected-device list and scan again if it is missing.")
            device_map[camera] = device_index
            continue
        parsed = urlparse(source)
        if parsed.scheme.lower() not in {"rtsp", "rtsps", "http", "https"} or not parsed.hostname:
            raise HTTPException(status_code=400, detail="Camera sources must be discovered USB devices or RTSP/RTSPS/HTTP/HTTPS stream URLs.")
        device_map[camera] = source
    if not device_map:
        raise HTTPException(status_code=400, detail="Select at least one USB camera or enter a network camera stream URL.")
    if len(set(device_map.values())) != len(device_map):
        raise HTTPException(status_code=400, detail="Each live camera view must use a different device.")
    with LIVE_SESSION_LOCK:
        if any(job.get("source_mode") in {"live_usb_camera", "live_camera"} and job.get("status") in {"initializing", "streaming", "stopping"} for job in JOBS.values()):
            raise HTTPException(status_code=409, detail="A live camera session is already running. Stop it before starting another.")
    # Keep the start request lightweight: probing CUDA imports torch and can
    # block the button response. The worker initializes the runtime and reports
    # detailed failures through the job status.
    missing = [name for name in ("torch", "ultralytics") if importlib.util.find_spec(name) is None]
    if missing:
        missing_text = ", ".join(missing)
        raise HTTPException(status_code=503, detail=f"Live analysis runtime is missing {missing_text}. Install requirements and restart the server.")

    job_id = uuid.uuid4().hex
    stop_event = threading.Event()
    job: dict[str, Any] = {
        "job_id": job_id,
        "source_mode": "live_camera",
        "status": "initializing",
        "message": "Opening camera sources",
        "performance_profile": performance_profile,
        "site_label": site_label,
        "camera_count": len(device_map),
        "camera_sources": {camera: ("network_stream" if isinstance(source, str) else "usb_camera") for camera, source in device_map.items()},
        "active_cameras": list(device_map),
        "camera_errors": {},
        "camera_states": {camera: "connecting" if camera in device_map else "not_connected" for camera in CAMERAS},
        "camera_progress": {camera: 0 for camera in CAMERAS},
        "live_counts_by_camera": {camera: {"vehicles_in_frame": 0, "pedestrians_in_frame": 0, "vehicle_tracks": 0, "pedestrian_tracks": 0} for camera in CAMERAS},
        "live_metrics_by_camera": {},
        "frame_sequences": {camera: 0 for camera in CAMERAS},
        "preview_jpegs": {},
        "live_identity": {"estimated_unique_vehicles": 0, "local_vehicle_tracklets": 0, "confirmed_cross_camera_matches": []},
        "decision": {"severity": "insufficient", "reason": "Connect all three views to produce a junction advisory."},
        "decision_timing": {},
        "decision_timeline": [{"elapsed_seconds": 0.0, "color": "yellow", "label": "YELLOW - COLLECTING EVIDENCE", "reason": "Initial cautious state while camera evidence is collected.", "triggered_rule": "initial_wait_state"}],
        "_decision_last_color": "yellow",
        "_decision_started_monotonic": time.monotonic(),
        "_stop_event": stop_event,
        "_update_lock": threading.Lock(),
        "_identity_associator": LiveCrossCameraAssociator(),
        "_identity_last_sample": {},
    }
    JOBS[job_id] = job
    task = asyncio.create_task(asyncio.to_thread(_run_live_job, job_id, device_map, performance_profile))
    RUNNING_TASKS.add(task)
    task.add_done_callback(RUNNING_TASKS.discard)
    return {"job_id": job_id, "status": "initializing", "camera_count": len(device_map)}


@router.post("/{job_id}/stop", tags=["analysis"])
def stop_analysis(job_id: str) -> dict[str, str]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    if job.get("status") not in {"queued", "initializing", "processing", "streaming", "stopping"}:
        return {"job_id": job_id, "status": str(job.get("status", "stopped"))}
    stop_event = job.get("_stop_event")
    if stop_event:
        stop_event.set()
    job["status"] = "stopping"
    job["stop_requested"] = True
    job["message"] = "Stopping after the current inference frame"
    return {"job_id": job_id, "status": "stopping"}


@router.get("/{job_id}")
def get_analysis_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found; it may have expired after a server restart.")
    internal_keys = {"preview_jpegs", "provisional_camera_metrics", "live_metrics_by_camera"}
    return {key: value for key, value in job.items() if key not in internal_keys and not key.startswith("_")}


@router.get("/{job_id}/frames/{camera_id}")
def get_latest_tracking_frame(job_id: str, camera_id: str) -> Response:
    if camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Unknown camera.")
    job = JOBS.get(job_id)
    frame = job.get("preview_jpegs", {}).get(camera_id) if job else None
    if frame is None:
        raise HTTPException(status_code=404, detail="No analyzed frame is ready yet.")
    return Response(content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{job_id}/report.pdf")
def download_analysis_report(job_id: str) -> Response:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Analysis report not found.")
    report_path = JOB_ROOT / job_id / "session_report.pdf"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="The PDF report is not ready or has expired.")
    # The UI downloads only after an explicit button click. Inline disposition
    # prevents a stale or stray navigation from silently downloading the PDF.
    return Response(content=report_path.read_bytes(), media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="traffic-analysis-{job_id[:12]}.pdf"', "Cache-Control": "no-store"})


def _analyze_camera(
    job_id: str,
    camera_index: int,
    camera_id: str,
    source_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    update_lock: threading.Lock,
    device: str,
    frame_stride: int,
    image_size: int,
    stop_event: threading.Event,
    decision_thresholds: DecisionThresholds,
    history_context: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Analyze one camera independently; called concurrently for each clip."""

    import cv2

    job = JOBS[job_id]
    camera_zones = _zones_for_analysis(camera_id, config["cameras"][camera_id])
    reference_width, reference_height = config.get("reference_resolution", [1280, 720])
    capture = open_video_capture(cv2, source_path)
    preview_ok, first_frame = capture.read()
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    capture.release()
    if min(frame_width, frame_height, reference_width, reference_height) <= 0:
        raise RuntimeError(f"Invalid frame or zone reference size for {camera_id}.")
    if preview_ok and first_frame is not None:
        preview_height, preview_width = first_frame.shape[:2]
        preview_scale = min(1.0, 640 / max(preview_width, preview_height))
        if preview_scale < 1:
            first_frame = cv2.resize(first_frame, (int(preview_width * preview_scale), int(preview_height * preview_scale)))
        first_frame = _draw_performance_hud(cv2, first_frame, [f"{camera_id.upper()} CLIP READY · LOADING TRACKING MODEL"])
        encoded, preview_buffer = cv2.imencode(".jpg", first_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if encoded:
            with update_lock:
                job["preview_jpegs"][camera_id] = preview_buffer.tobytes()
                job["frame_sequences"][camera_id] += 1
    if 0 < source_fps < 1000:
        target_sample_fps = TARGET_FPS if device != "cpu" else (10 if frame_stride >= 3 else 15)
        frame_stride = max(frame_stride, round(source_fps / target_sample_fps))
    scale_x, scale_y = frame_width / reference_width, frame_height / reference_height
    scaled_zone_data = [
        {**zone, "polygon": [[round(point[0] * scale_x), round(point[1] * scale_y)] for point in zone["polygon"]]}
        for zone in camera_zones
    ]
    max_x = max((point[0] for zone in scaled_zone_data for point in zone["polygon"]), default=0)
    max_y = max((point[1] for zone in scaled_zone_data for point in zone["polygon"]), default=0)
    fit_scale = min(1.0, (frame_width - 1) / max_x, (frame_height - 1) / max_y) if scaled_zone_data else 1.0
    if fit_scale < 1.0:
        for zone in scaled_zone_data:
            zone["polygon"] = [[round(x * fit_scale), round(y * fit_scale)] for x, y in zone["polygon"]]
    LOGGER.info("Analyzing %s (%sx%s) concurrently", camera_id, frame_width, frame_height)

    def progress(done: int, total: int) -> None:
        with update_lock:
            if total:
                job["camera_progress"][camera_id] = min(99, round(done / total * 100))
            else:
                job["camera_progress"][camera_id] = min(99, int(job["camera_progress"].get(camera_id, 0)))
            job["progress_percent"] = round(sum(job["camera_progress"].values()) / len(CAMERAS))
            job["frames_processed_by_camera"] = {
                **job.get("frames_processed_by_camera", {}), camera_id: done,
            }

    def publish_metrics(metrics: dict[str, Any]) -> None:
        metrics.update({
            "camera_id": camera_id,
            "roi_configuration_valid": valid_zone_configuration(camera_id, camera_zones),
            "vehicle_track_count": int(metrics.get("relevant_vehicle_track_count", 0)),
            "source_mode": "recorded",
            "updated_at_epoch": time.time(),
        })
        with update_lock:
            job["provisional_camera_metrics"][camera_id] = metrics
            now = time.monotonic()
            conflict_seen = (
                metrics.get("pedestrian_in_free_left_route_now") is True
                or metrics.get("pedestrian_vehicle_overlap_now") is True
                or metrics.get("turn_path_blocked_now") is True
            )
            if conflict_seen or now - job.get("_last_decision_update_monotonic", 0.0) >= 0.2:
                _publish_job_decision(job, recommend(
                    list(job["provisional_camera_metrics"].values()),
                    decision_thresholds,
                    history_context,
                ), now)
                job["_last_decision_update_monotonic"] = now

    def preview(frame: Any, counts: dict[str, Any]) -> None:
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        if scale < 1:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if encoded:
            with update_lock:
                job["preview_jpegs"][camera_id] = buffer.tobytes()
                job["frame_sequences"][camera_id] += 1
                job["live_counts_by_camera"][camera_id] = counts
                job["camera_states"][camera_id] = "analyzing"
                active_count = sum(value == "analyzing" for value in job["camera_states"].values())
                job["message"] = f"Analyzing {active_count}/{len(CAMERAS)} camera streams"

    model_load_started = time.perf_counter()
    detector = UltralyticsByteTrackDetector(
        image_size=image_size,
        device=device,
        classes=None if camera_id == "turning" else TRAFFIC_CLASS_IDS,
    )
    model_load_seconds = round(time.perf_counter() - model_load_started, 2)
    with update_lock:
        job["camera_model_load_seconds"][camera_id] = model_load_seconds
        job["camera_states"] = {**job.get("camera_states", {}), camera_id: "model_ready"}
        ready_count = sum(value == "model_ready" for value in job["camera_states"].values())
        job["message"] = f"Detection models ready for {ready_count}/{len(CAMERAS)} cameras"

    summary = analyze_video(
        camera_id=camera_id,
        source_path=source_path,
        output_path=output_dir / f"{camera_id}_annotated.mp4",
        zones=zones_from_mapping(scaled_zone_data),
        detector=detector,
        waiting_threshold_s=float(config.get("thresholds", {}).get("waiting_seconds", 10)),
        blockage_threshold_s=float(config.get("thresholds", {}).get("blockage_start_seconds", 5)),
        blockage_medium_s=float(config.get("thresholds", {}).get("blockage_medium_seconds", 10)),
        blockage_high_s=float(config.get("thresholds", {}).get("blockage_high_seconds", 30)),
        stationary_speed_px_s=float(config.get("thresholds", {}).get("stationary_speed_px_per_second", 8)) * (scale_x * scale_y) ** 0.5,
        pedestrian_clearance_hold_s=float(config.get("thresholds", {}).get("pedestrian_clearance_hold_seconds", 2)),
        progress_callback=progress,
        preview_callback=preview,
        metrics_callback=publish_metrics,
        save_annotated_video=False,
        frame_stride=frame_stride,
        identity_time_offset_s=float(config.get("camera_clock_offsets_seconds", {}).get(camera_id, 0.0)),
        stop_event=stop_event,
    )
    summary["display_name"] = {"approach": "Road 1 · main approach / free-left", "turning": "Road 2 · free-left / pedestrian conflict", "exit": "Road 3 · right-side joining road"}[camera_id]
    summary["analysis_image_size"] = image_size
    summary["model_load_seconds"] = model_load_seconds
    summary["roi_configuration_valid"] = valid_zone_configuration(camera_id, camera_zones)
    summary["configured_zone_kinds"] = sorted({str(zone.get("kind", "")) for zone in camera_zones})
    with update_lock:
        job["camera_progress"][camera_id] = 100
        job["progress_percent"] = round(sum(job["camera_progress"].values()) / len(CAMERAS))
    return camera_index, summary


def _finalize_early_stopped_job(
    job_id: str,
    job: dict[str, Any],
    output_dir: Path,
    site_label: str,
    performance_profile: str,
    message: str,
) -> None:
    """Persist a useful partial PDF even when stop arrives before the first frame."""

    result: dict[str, Any] = {
        "session_mode": "recorded",
        "performance_profile": performance_profile,
        "site_label": site_label,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_count": 0,
        "data_status": "partial",
        "camera_results": [],
        "cross_camera_identity": {},
        "historical_context": _site_history(site_label),
        "junction_decision": job.get("decision", {"severity": "insufficient", "reason": message}),
        "decision_timing": dict(job.get("decision_timing", {})),
        "decision_timeline": list(job.get("decision_timeline", [])),
        "limitations": [message, "No camera frames were available, so no traffic measurements were produced."],
    }
    try:
        generate_session_pdf(result, output_dir / "session_report.pdf")
        result["report_url"] = f"/api/v1/analysis/{job_id}/report.pdf"
    except Exception as error:
        LOGGER.exception("Could not create early-stop report for %s", job_id)
        result["report_error"] = str(error)
    try:
        (output_dir / "session_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except OSError:
        LOGGER.exception("Could not save early-stop summary for %s", job_id)
    job.update(status="stopped", message=message, result=result, junction_decision=result["junction_decision"])


def _run_job(job_id: str, input_dir: Path, paths: list[Path], performance_profile: str = "fast", site_label: str = DEFAULT_SITE_LABEL, stop_event: threading.Event | None = None) -> None:
    job = JOBS[job_id]
    stop_event = stop_event or threading.Event()
    output_dir = JOB_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if stop_event.is_set():
            _finalize_early_stopped_job(job_id, job, output_dir, site_label, performance_profile, "Analysis stopped before processing began.")
            return
        job.update(status="initializing", message="Checking the detection runtime")
        config = json.loads(ZONE_CONFIG.read_text(encoding="utf-8"))
        _validate_zone_config(config)
        history_context = _site_history(site_label)
        decision_thresholds = _decision_thresholds(config)
        job["historical_context"] = history_context
        import cv2
        import torch

        torch_threads = max(1, (os.cpu_count() or 3) // len(CAMERAS))
        torch.set_num_threads(torch_threads)
        cv2.setNumThreads(1)
        device = "0" if torch.cuda.is_available() else "cpu"
        if device != "cpu":
            torch.backends.cudnn.benchmark = True
        if performance_profile == "fast":
            # CPU inference is the bottleneck on the reported run. Sample at
            # about 10 FPS and use a smaller inference size in fast mode; this still
            # gives multiple observations for ordinary moving objects.
            frame_stride = 1 if device == "0" else 3
            image_sizes = {
                camera: (512 if camera == "turning" else 416) if device == "0" else 320
                for camera in CAMERAS
            }
        else:
            # Per-camera FPS metadata below keeps balanced mode near 15 CPU
            # samples/sec, avoiding redundant work without over-skipping
            # lower-frame-rate camera clips.
            frame_stride = 1 if device == "0" else 2
            image_sizes = {
                camera: (640 if camera == "turning" else 512) if device == "0" else (512 if camera == "turning" else 416)
                for camera in CAMERAS
            }
        job.update(
            status="processing",
            message=f"Loading detection models · {performance_profile} mode",
            performance_profile=performance_profile,
            camera_states={camera: "loading_model" for camera in CAMERAS},
            compute_device="GPU 0" if device == "0" else "CPU",
            cpu_threads_per_camera=torch_threads,
            frame_stride=frame_stride,
            image_sizes=image_sizes,
        )
        LOGGER.info("Analysis job %s: device=%s, torch_threads_per_camera=%s, frame_stride=%s, image_sizes=%s", job_id, device, torch_threads, frame_stride, image_sizes)
        if stop_event.is_set():
            _finalize_early_stopped_job(job_id, job, output_dir, site_label, performance_profile, "Analysis stopped before any video frames were processed.")
            return
        summaries: list[dict[str, Any] | None] = [None] * len(CAMERAS)
        camera_errors: dict[str, str] = {}
        update_lock: threading.Lock = job["_update_lock"]
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="traffic-camera") as executor:
            futures = {
                executor.submit(_analyze_camera, job_id, index, camera_id, source_path, output_dir, config, update_lock, device, frame_stride, image_sizes[camera_id], stop_event, decision_thresholds, history_context): (index, camera_id)
                for index, (camera_id, source_path) in enumerate(zip(CAMERAS, paths))
            }
            for future in as_completed(futures):
                index, camera_id = futures[future]
                try:
                    _, summary = future.result()
                    summaries[index] = summary
                except Exception as error:
                    LOGGER.exception("Recorded camera %s failed in session %s", camera_id, job_id)
                    camera_errors[camera_id] = str(error)
                    summaries[index] = {
                        "camera_id": camera_id,
                        "display_name": {"approach": "Road 1 · main approach / free-left", "turning": "Road 2 · free-left / pedestrian conflict", "exit": "Road 3 · right-side joining road"}[camera_id],
                        "status": "error",
                        "error": str(error),
                        "frame_count_processed": 0,
                        "frame_coverage_percent": 0,
                    }
                    job.setdefault("camera_states", {})[camera_id] = "error"
                    job.setdefault("camera_errors", {})[camera_id] = str(error)

        camera_summaries = [summary for summary in summaries if summary is not None]
        valid_camera_summaries = [item for item in camera_summaries if item.get("status") != "error"]
        camera_warnings = {
            str(item["camera_id"]): list(item.get("input_warnings", []))
            for item in valid_camera_summaries if item.get("input_warnings")
        }
        incomplete_coverage = any(
            item.get("frame_coverage_percent") is not None and item.get("frame_coverage_percent") < 100
            for item in valid_camera_summaries
        )
        for item in valid_camera_summaries:
            item["vehicle_track_count"] = int(item.get("relevant_vehicle_track_count", 0))
        cross_camera = associate_camera_tracks(valid_camera_summaries)
        decision = recommend(camera_summaries, decision_thresholds, history_context, cross_camera)
        _publish_job_decision(job, decision)
        result = {
            "session_mode": "recorded",
            "performance_profile": performance_profile,
            "site_label": site_label,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "camera_count": len(valid_camera_summaries),
            "data_status": "partial" if stop_event.is_set() or camera_errors or incomplete_coverage or camera_warnings else "complete",
            "camera_results": camera_summaries,
            "cross_camera_identity": cross_camera,
            "historical_context": history_context,
            "junction_decision": decision,
            "decision_timing": dict(job.get("decision_timing", {})),
            "decision_timeline": list(job.get("decision_timeline", [])),
            "camera_errors": camera_errors,
            "camera_warnings": camera_warnings,
            "limitations": ["A failed or unreadable clip is reported per camera and causes an insufficient-data junction decision; remaining camera analyses and the partial PDF are retained.", "Cross-camera matches are conservative appearance matches, not ground-truth identities; synchronized clips are required and uncertain matches remain separate.", "Queue extent is image-plane pixels; physical length and calibrated speed require a road-plane calibration and stop-line location.", "Example camera zones must be calibrated to actual road views; the advisory does not operate a traffic signal."],
        }
        generate_session_pdf(result, output_dir / "session_report.pdf")
        result["report_url"] = f"/api/v1/analysis/{job_id}/report.pdf"
        (output_dir / "session_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        final_status = "stopped" if stop_event.is_set() else "partial" if camera_errors or incomplete_coverage or camera_warnings else "complete"
        partial_reasons = len(camera_errors) + len(camera_warnings)
        message = "Analysis stopped; partial evidence is shown." if stop_event.is_set() else f"Partial analysis: {partial_reasons} camera clip(s) had input or decoding issues; a report is ready." if final_status == "partial" else "Analysis complete"
        job.update(status=final_status, message=message, progress_percent=round(sum(job.get("camera_progress", {}).values()) / len(CAMERAS)), result=result, junction_decision=decision, decision=decision)
    except Exception as error:
        LOGGER.exception("Recorded analysis job %s failed", job_id)
        job.update(status="stopped" if stop_event.is_set() else "failed", message="Analysis stopped." if stop_event.is_set() else "Analysis failed", error=str(error))
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)


def _run_live_job(job_id: str, device_map: dict[str, int | str], performance_profile: str) -> None:
    """Run independent latest-frame camera workers and update a live advisory."""

    job = JOBS[job_id]
    stop_event: threading.Event = job["_stop_event"]
    try:
        import cv2
        import torch
    except Exception as error:
        LOGGER.exception("Live inference runtime failed to import")
        stop_event.set()
        job.update(status="failed", error=str(error), message=f"Could not start live inference: {error}")
        job.pop("_identity_associator", None)
        job.pop("_identity_last_sample", None)
        return
    update_lock: threading.Lock = job["_update_lock"]
    try:
        config = json.loads(ZONE_CONFIG.read_text(encoding="utf-8"))
        _validate_zone_config(config)
        cpu_threads = max(1, (os.cpu_count() or 3) // max(1, len(device_map)))
        torch.set_num_threads(cpu_threads)
        cv2.setNumThreads(1)
        device = "0" if torch.cuda.is_available() else "cpu"
        if device != "cpu":
            torch.backends.cudnn.benchmark = True
        if performance_profile == "fast":
            image_size = 512 if device == "0" else 416
        else:
            image_size = 640 if device == "0" else 512
    except Exception as error:
        LOGGER.exception("Could not initialize live analysis settings")
        stop_event.set()
        job.update(status="failed", error=str(error), message=f"Could not initialize live analysis: {error}")
        job.pop("_identity_associator", None)
        job.pop("_identity_last_sample", None)
        return
    site_label = str(job.get("site_label", DEFAULT_SITE_LABEL))
    history_context = _site_history(site_label)
    decision_thresholds = _decision_thresholds(config)
    job["historical_context"] = history_context
    source_started_at = time.perf_counter()
    job.update(status="streaming", message="Connecting to camera sources", compute_device="GPU 0" if device == "0" else "CPU", image_size=image_size, camera_states={camera: "connecting" if camera in device_map else "not_connected" for camera in CAMERAS})

    def on_state(camera_id: str, state: str, model_load_seconds: float | None) -> None:
        with update_lock:
            job["camera_states"][camera_id] = state
            if model_load_seconds is not None:
                job.setdefault("camera_model_load_seconds", {})[camera_id] = round(model_load_seconds, 2)
            ready = sum(value == "analyzing" for value in job["camera_states"].values())
            job["message"] = f"Live tracking · {ready}/{len(device_map)} cameras analyzing"

    def on_update(camera_id: str, frame: Any, counts: dict[str, Any], tracklets: list[dict[str, Any]], metrics: dict[str, Any] | None) -> None:
        if metrics is not None:
            metrics["camera_id"] = camera_id
            metrics["roi_configuration_valid"] = valid_zone_configuration(camera_id, config["cameras"].get(camera_id, []))
            metrics["configured_zone_kinds"] = sorted({str(zone.get("kind", "")) for zone in config["cameras"].get(camera_id, [])})
            metrics["source_mode"] = "live"
            metrics["updated_at_epoch"] = time.time()
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        if scale < 1:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        with update_lock:
            if metrics is not None:
                identity: LiveCrossCameraAssociator = job["_identity_associator"]
                last_samples: dict[tuple[str, int], float] = job["_identity_last_sample"]
                for track in tracklets:
                    key = (camera_id, int(track["track_id"]))
                    sample_time = float(track.get("end_s", 0))
                    if int(track.get("samples", 0)) >= 2 and sample_time > last_samples.get(key, -1.0):
                        last_samples[key] = sample_time
                        identity.observe(camera_id, track)
                identity_summary = identity.summary()
                local_vehicle_count = sum(
                    sum(int(value) for value in camera_metrics.get("vehicle_category_tracks", {}).values())
                    for camera_metrics in {**job.get("live_metrics_by_camera", {}), camera_id: metrics}.values()
                )
                accepted_duplicate_count = identity_summary["local_vehicle_tracklets"] - identity_summary["estimated_unique_vehicles"]
                identity_summary["estimated_unique_vehicles"] = max(0, local_vehicle_count - accepted_duplicate_count)
                identity_summary["unique_vehicle_count_upper_bound"] = local_vehicle_count
                job.setdefault("live_metrics_by_camera", {})[camera_id] = metrics
                job["live_identity"] = identity_summary
                active_metrics = [job["live_metrics_by_camera"][cam] for cam in CAMERAS if cam in job["live_metrics_by_camera"]]
                now = time.monotonic()
                conflict_seen = (
                    metrics.get("pedestrian_in_free_left_route_now") is True
                    or metrics.get("pedestrian_vehicle_overlap_now") is True
                    or metrics.get("turn_path_blocked_now") is True
                )
                if conflict_seen or now - job.get("_last_decision_update_monotonic", 0.0) >= 0.2:
                    _publish_job_decision(job, recommend(active_metrics, decision_thresholds, history_context, identity_summary), now)
                    job["_last_decision_update_monotonic"] = now
            identity_summary = job.get("live_identity", {})
            capture_fps = counts.get("source_fps", 0)
            capture_fps_text = f"{capture_fps:.1f}" if isinstance(capture_fps, (int, float)) and capture_fps > 0 else "—"
            frame = _draw_performance_hud(cv2, frame, [
                f"ANALYSIS {counts.get('inference_fps', 0):.1f} FPS  |  CAPTURE {capture_fps_text} FPS",
                f"FRAME AGE {counts.get('frame_age_ms', '—')} ms  |  DROPPED {counts.get('dropped_capture_frames', 0)}",
            ])
            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not encoded:
                return
            job["preview_jpegs"][camera_id] = buffer.tobytes()
            job["frame_sequences"][camera_id] += 1
            job["live_counts_by_camera"][camera_id] = counts

    def on_preview(camera_id: str, frame: Any) -> None:
        """Publish a first raw frame while the per-camera detector is loading."""
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        if scale < 1:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        frame = _draw_performance_hud(cv2, frame.copy(), ["CAMERA CONNECTED  |  LOADING TRACKING MODEL"])
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not encoded:
            return
        with update_lock:
            job["preview_jpegs"][camera_id] = buffer.tobytes()
            job["frame_sequences"][camera_id] += 1

    errors: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=len(device_map), thread_name_prefix="live-usb-camera") as executor:
            futures = {
                executor.submit(
                    run_live_camera,
                    camera_id=camera_id,
                    device_index=device_index,
                    source_started_at=source_started_at,
                    stop_event=stop_event,
                    config=config,
                    device=device,
                    image_size=image_size,
                    identity_time_offset_s=float(config.get("camera_clock_offsets_seconds", {}).get(camera_id, 0.0)),
                    on_state=lambda state, load, cam=camera_id: on_state(cam, state, load),
                    on_preview=lambda frame, cam=camera_id: on_preview(cam, frame),
                    on_update=lambda frame, counts, tracks, metrics, cam=camera_id: on_update(cam, frame, counts, tracks, metrics),
                ): camera_id
                for camera_id, device_index in device_map.items()
            }
            for future in as_completed(futures):
                camera_id = futures[future]
                try:
                    summaries[camera_id] = future.result()
                    summaries[camera_id]["display_name"] = {"approach": "Road 1 · main approach / free-left", "turning": "Road 2 · free-left / pedestrian conflict", "exit": "Road 3 · right-side joining road"}[camera_id]
                    summaries[camera_id]["roi_configuration_valid"] = valid_zone_configuration(camera_id, config["cameras"].get(camera_id, []))
                    summaries[camera_id]["configured_zone_kinds"] = sorted({str(zone.get("kind", "")) for zone in config["cameras"].get(camera_id, [])})
                except Exception as error:
                    LOGGER.exception("Live camera %s failed in session %s", camera_id, job_id)
                    errors.append(f"{camera_id}: {error}")
                    job["camera_states"][camera_id] = "error"
                    job.setdefault("camera_errors", {})[camera_id] = str(error)
                    job["error"] = str(error)
    except Exception as error:
        LOGGER.exception("Live analysis session %s failed", job_id)
        errors.append(str(error))
    finally:
        for camera_id in device_map:
            if job.get("camera_states", {}).get(camera_id) not in {"error", "stopped"}:
                job["camera_states"][camera_id] = "stopped"
        active_metrics = [job.get("live_metrics_by_camera", {}).get(cam) for cam in CAMERAS]
        active_metrics = [metrics for metrics in active_metrics if metrics is not None]
        job["camera_results"] = list(summaries.values())
        decision_metrics = active_metrics if not errors and len(device_map) == len(CAMERAS) and len(summaries) == len(CAMERAS) else []
        job["junction_decision"] = recommend(decision_metrics, decision_thresholds, history_context, job.get("live_identity", {}))
        _publish_job_decision(job, job["junction_decision"])
        result = {
            "session_mode": "live_camera",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "site_label": site_label,
            "camera_count": len(summaries),
            "data_status": "complete" if len(summaries) == len(CAMERAS) else "partial",
            "camera_errors": dict(job.get("camera_errors", {})),
            "camera_results": list(summaries.values()),
            "historical_context": history_context,
            "junction_decision": job["junction_decision"],
            "decision_timing": dict(job.get("decision_timing", {})),
            "decision_timeline": list(job.get("decision_timeline", [])),
            "limitations": ["Live reports contain measurements collected before stop; a failed or missing camera makes the junction advisory incomplete.", "Camera-local tracking and appearance-based cross-camera matches are estimates, not ground-truth identities.", "The advisory is for human review and does not control traffic signals."],
        }
        try:
            output_dir = JOB_ROOT / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            generate_session_pdf(result, output_dir / "session_report.pdf")
            result["report_url"] = f"/api/v1/analysis/{job_id}/report.pdf"
            (output_dir / "session_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception as error:
            LOGGER.exception("Could not create live report for %s", job_id)
            result["report_error"] = str(error)
        job["result"] = result
        job["status"] = "failed" if errors else "stopped"
        job["message"] = "; ".join(errors) if errors else "Live camera analysis stopped"
        job.pop("_identity_associator", None)
        job.pop("_identity_last_sample", None)
