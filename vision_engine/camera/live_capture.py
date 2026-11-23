"""Low-latency capture helpers for locally attached USB/UVC cameras."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from vision_engine.camera.video_analysis import _draw_overlay
from vision_engine.detection.yolo import Detector, UltralyticsByteTrackDetector
from vision_engine.metrics.traffic import TrafficAnalyzer, VEHICLE_CLASSES
from vision_engine.roi.zones import Zone, zones_from_mapping

TARGET_FPS = 30


class LatestFrameSource:
    """Continuously drain a webcam and retain only its newest frame."""

    def __init__(self, device_index: int, stop_event: threading.Event) -> None:
        import cv2

        self.cv2 = cv2
        self.stop_event = stop_event
        self.capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(device_index, cv2.CAP_MSMF)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(device_index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open local camera device {device_index}. Enable the phone's USB webcam mode and check Windows camera access.")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.lock = threading.Lock()
        self.latest: tuple[Any, int, float] | None = None
        self.error: str | None = None
        self.sequence = 0
        self.thread = threading.Thread(target=self._read_loop, name=f"usb-camera-{device_index}", daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        failures = 0
        while not self.stop_event.is_set():
            try:
                ok, frame = self.capture.read()
            except Exception as error:
                self.error = f"Camera capture failed: {error}"
                return
            if not ok:
                failures += 1
                if failures >= 50:
                    self.error = "Camera stopped sending frames. Check the USB cable and phone webcam mode."
                    return
                time.sleep(0.01)
                continue
            failures = 0
            captured_at = time.perf_counter()
            with self.lock:
                self.sequence += 1
                self.latest = (frame, self.sequence, captured_at)

    def get_newer(self, after_sequence: int) -> tuple[Any, int, float] | None:
        with self.lock:
            if self.latest is None or self.latest[1] <= after_sequence:
                return None
            return self.latest

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.capture.release()


def enumerate_usb_cameras(max_devices: int = 8) -> list[dict[str, int]]:
    """Probe local Windows camera indices and return only devices that yield a frame."""

    import cv2

    found: list[dict[str, int]] = []
    for device_index in range(max_devices):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            capture = cv2.VideoCapture(device_index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = capture.read()
                if ok and frame is not None:
                    height, width = frame.shape[:2]
                    found.append({"device_index": device_index, "width": int(width), "height": int(height)})
                    break
            finally:
                capture.release()
    return found


def run_live_camera(
    *,
    camera_id: str,
    device_index: int,
    source_started_at: float,
    stop_event: threading.Event,
    config: dict[str, Any],
    device: str,
    image_size: int,
    identity_time_offset_s: float,
    on_state: Callable[[str, float | None], None],
    on_update: Callable[[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any] | None], None],
) -> dict[str, Any]:
    """Infer on the newest USB frame, dropping stale capture frames by design."""

    source = LatestFrameSource(device_index, stop_event)
    try:
        on_state("loading_model", None)
        model_started = time.perf_counter()
        detector: Detector = UltralyticsByteTrackDetector(image_size=image_size, device=device)
        model_load_seconds = time.perf_counter() - model_started
        on_state("waiting_for_frame", model_load_seconds)
        first_frame: tuple[Any, int, float] | None = None
        wait_until = time.perf_counter() + 5.0
        while first_frame is None and time.perf_counter() < wait_until and not stop_event.is_set():
            first_frame = source.get_newer(-1)
            if first_frame is None:
                if source.error:
                    raise RuntimeError(source.error)
                time.sleep(0.02)
        if first_frame is None:
            if stop_event.is_set():
                return {"camera_id": camera_id, "device_index": device_index, "status": "stopped", "inference_frames": 0}
            raise RuntimeError(f"No frames arrived from camera device {device_index}.")

        frame, _, _ = first_frame
        height, width = frame.shape[:2]
        if width < 2 or height < 2:
            raise RuntimeError(f"Camera device {device_index} returned an invalid {width}x{height} frame.")
        reference_width, reference_height = config.get("reference_resolution", [1280, 720])
        if len(config.get("reference_resolution", [])) != 2 or reference_width <= 0 or reference_height <= 0:
            raise ValueError("Camera zone reference_resolution must contain two positive dimensions.")
        scale_x, scale_y = width / reference_width, height / reference_height
        raw_zones = config["cameras"][camera_id]
        if not raw_zones or any(len(zone.get("polygon", [])) < 3 for zone in raw_zones):
            raise ValueError(f"Camera zone configuration for {camera_id} must contain polygons with at least three points.")
        scaled = [
            {**zone, "polygon": [[round(point[0] * scale_x), round(point[1] * scale_y)] for point in zone["polygon"]]}
            for zone in raw_zones
        ]
        max_x = max(point[0] for zone in scaled for point in zone["polygon"])
        max_y = max(point[1] for zone in scaled for point in zone["polygon"])
        fit_scale = min(1.0, (width - 1) / max(max_x, 1), (height - 1) / max(max_y, 1))
        if fit_scale < 1.0:
            for zone in scaled:
                zone["polygon"] = [[round(x * fit_scale), round(y * fit_scale)] for x, y in zone["polygon"]]
        zones = zones_from_mapping(scaled)
        thresholds = config.get("thresholds", {})
        analyzer = TrafficAnalyzer(
            camera_id, zones,
            stationary_speed_px_s=float(thresholds.get("stationary_speed_px_per_second", 8)) * (scale_x * scale_y) ** 0.5,
            waiting_threshold_s=float(thresholds.get("waiting_seconds", 10)),
            blockage_threshold_s=float(thresholds.get("blockage_start_seconds", 5)),
            blockage_medium_s=float(thresholds.get("blockage_medium_seconds", 10)),
            blockage_high_s=float(thresholds.get("blockage_high_seconds", 30)),
            pedestrian_clearance_hold_s=float(thresholds.get("pedestrian_clearance_hold_seconds", 2)),
        )
        on_state("analyzing", model_load_seconds)
        last_sequence = -1
        last_feature_sample: dict[int, float] = {}
        tracklets: dict[int, dict[str, Any]] = {}
        inference_frames = 0
        inference_seconds = 0.0
        loop_started = time.perf_counter()
        last_inference_at = 0.0
        last_preview_at = 0.0
        last_metrics_at = 0.0
        while not stop_event.is_set():
            item = source.get_newer(last_sequence)
            if item is None:
                if source.error:
                    raise RuntimeError(source.error)
                time.sleep(0.003)
                continue
            inference_started = time.perf_counter()
            wait_s = (1 / TARGET_FPS) - (inference_started - last_inference_at) if last_inference_at else 0.0
            if wait_s > 0:
                time.sleep(wait_s)
                continue
            frame, sequence, captured_at = item
            last_sequence = sequence
            inference_started = time.perf_counter()
            last_inference_at = inference_started
            detections = detector.detect_and_track(frame)
            inference_seconds += time.perf_counter() - inference_started
            inference_frames += 1
            timestamp_s = max(0.0, captured_at - source_started_at)
            labels = analyzer.update(detections, timestamp_s)
            vehicle_boxes = [d for d in detections if d.category.lower() in VEHICLE_CLASSES]
            pedestrians = sum(d.category.lower() == "person" for d in detections)
            for detection in vehicle_boxes:
                if detection.track_id is None or timestamp_s - last_feature_sample.get(detection.track_id, -1e9) < 0.5:
                    continue
                last_feature_sample[detection.track_id] = timestamp_s
                x1, y1, x2, y2 = (int(v) for v in detection.box)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue
                hsv = self_hsv_crop(source.cv2, frame[y1:y2, x1:x2])
                hist = source.cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
                source.cv2.normalize(hist, hist, alpha=1.0, norm_type=source.cv2.NORM_L2)
                feature = hist.flatten().astype(float).tolist()
                old = tracklets.get(detection.track_id)
                identity_time = timestamp_s + identity_time_offset_s
                if old is None:
                    tracklets[detection.track_id] = {"track_id": detection.track_id, "category": detection.category.lower(), "start_s": identity_time, "end_s": identity_time, "samples": 1, "color_hist": feature}
                else:
                    n = old["samples"]
                    old["color_hist"] = [(a * n + b) / (n + 1) for a, b in zip(old["color_hist"], feature)]
                    old["samples"] = n + 1
                    old["end_s"] = identity_time

            elapsed = max(time.perf_counter() - loop_started, 0.001)
            age_ms = max(0.0, (time.perf_counter() - captured_at) * 1000)
            counts = {
                "vehicles_in_frame": len(vehicle_boxes),
                "pedestrians_in_frame": pedestrians,
                "pedestrians_in_free_left_route": analyzer.metrics.pedestrian_in_free_left_route_count_now,
                "vehicle_tracks": sum(analyzer.metrics.category_counts.values()),
                "pedestrian_tracks": analyzer.metrics.pedestrian_tracks,
                "processing_fps": round(inference_frames / elapsed, 2),
                "inference_fps": round(inference_frames / max(inference_seconds, 0.001), 2),
                "source_fps": round(float(source.capture.get(source.cv2.CAP_PROP_FPS) or 0), 2),
                "frame_age_ms": round(age_ms),
                "dropped_capture_frames": max(0, sequence - inference_frames),
                "inference_stride": "latest-frame",
                "model_load_seconds": round(model_load_seconds, 2),
                "joining_road_moving_vehicles": analyzer.metrics.joining_flow_last_moving_vehicles,
                "joining_road_stationary_vehicles": analyzer.metrics.joining_flow_last_stationary_vehicles,
                "joining_road_unclassified_vehicles": analyzer.metrics.joining_flow_last_unclassified_vehicles,
                "joining_road_vehicles": analyzer.metrics.joining_flow_last_vehicles,
            }
            track_summaries = list(tracklets.values())
            now = time.perf_counter()
            metric_snapshot = None
            if now - last_metrics_at >= 0.2 or inference_frames == 1:
                metric_snapshot = analyzer.metrics.to_dict()
                metric_snapshot["total_stopped_delay_seconds"] = round(sum(s.stopped_delay_s for s in analyzer.states.values()), 2)
                metric_snapshot["maximum_blockage_duration_seconds"] = round(max(
                    [float(event.get("duration_seconds", 0)) for event in analyzer.metrics.blockage_events]
                    + [max(0.0, timestamp_s - started) for started in analyzer.open_blockages.values()]
                    + [0.0]
                ), 2)
                metric_snapshot["maximum_queue_extent_pixels"] = metric_snapshot.get("maximum_queue_extent_pixels", 0)
                last_metrics_at = now
            if now - last_preview_at >= 1 / TARGET_FPS or inference_frames == 1:
                annotated = _draw_overlay(source.cv2, frame.copy(), zones, detections, labels, sequence, 0)
                from vision_engine.camera.video_analysis import _draw_performance_hud
                capture_fps = counts["source_fps"]
                capture_fps_text = f"{capture_fps:.1f}" if capture_fps > 0 else "—"
                annotated = _draw_performance_hud(source.cv2, annotated, [
                    f"ANALYSIS {counts['inference_fps']:.1f} FPS / TARGET {TARGET_FPS}  |  CAPTURE {capture_fps_text} FPS",
                    f"FRAME AGE {counts['frame_age_ms']} ms  |  DROPPED {counts['dropped_capture_frames']}",
                ])
                on_update(annotated, counts, track_summaries, metric_snapshot)
                last_preview_at = now
        duration = max(0.0, time.perf_counter() - source_started_at)
        summary = analyzer.finish(duration)
        summary.update({"camera_id": camera_id, "device_index": device_index, "model_load_seconds": round(model_load_seconds, 2), "inference_frames": inference_frames, "inference_fps": round(inference_frames / max(inference_seconds, 0.001), 2), "end_to_end_fps": round(inference_frames / max(time.perf_counter() - loop_started, 0.001), 2)})
        return summary
    finally:
        source.close()


def self_hsv_crop(cv2: Any, crop: Any) -> Any:
    """Convert a crop for the in-memory appearance histogram."""

    return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
