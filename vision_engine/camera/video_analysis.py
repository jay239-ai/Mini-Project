"""Frame-by-frame recorded video analysis and annotation output."""

import time
import threading
import math
from pathlib import Path
from typing import Any
from collections.abc import Callable

from vision_engine.detection.yolo import Detector
from vision_engine.metrics.traffic import TrafficAnalyzer, VEHICLE_CLASSES, is_rider_detection
from vision_engine.roi.zones import Zone

TARGET_FPS = 30


def open_video_capture(cv2: Any, source_path: Path) -> Any:
    """Prefer FFmpeg for uploaded files, then fall back to OpenCV's platform backend."""

    try:
        capture = cv2.VideoCapture(str(source_path), cv2.CAP_FFMPEG)
    except (AttributeError, TypeError, cv2.error):
        capture = cv2.VideoCapture(str(source_path))
    if capture.isOpened():
        return capture
    capture.release()
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video file: {source_path}")
    return capture


def analyze_video(
    camera_id: str,
    source_path: Path,
    output_path: Path,
    zones: list[Zone],
    detector: Detector,
    waiting_threshold_s: float = 10.0,
    blockage_threshold_s: float = 5.0,
    blockage_medium_s: float = 10.0,
    blockage_high_s: float = 30.0,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_callback: Callable[[Any, dict[str, Any]], None] | None = None,
    metrics_callback: Callable[[dict[str, Any]], None] | None = None,
    save_annotated_video: bool = True,
    frame_stride: int = 1,
    identity_time_offset_s: float = 0.0,
    stop_event: threading.Event | None = None,
    stationary_speed_px_s: float = 8.0,
    pedestrian_clearance_hold_s: float = 2.0,
) -> dict[str, Any]:
    """Analyze each selected frame, optionally writing a clear annotated video plus metrics."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to read and annotate video") from error

    if not source_path.is_file():
        raise FileNotFoundError(f"Video file not found: {source_path}")
    if frame_stride < 1:
        raise ValueError("Frame stride must be at least one")
    if save_annotated_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = open_video_capture(cv2, source_path)
    try:
        capture_backend = capture.getBackendName()
    except (AttributeError, cv2.error):
        capture_backend = "unknown"

    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    input_warnings: list[str] = []
    fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 30.0
    if fps != reported_fps:
        input_warnings.append("The clip did not report a usable frame rate; timestamps use a 30 FPS fallback where media timestamps are unavailable.")
    reported_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = int(reported_frame_count) if math.isfinite(reported_frame_count) and reported_frame_count > 0 else 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        capture.release()
        raise RuntimeError(f"Video does not report a valid frame size: {source_path}")

    for zone in zones:
        for x, y in zone.polygon:
            if not (0 <= x < width and 0 <= y < height):
                capture.release()
                raise ValueError(f"Zone '{zone.name}' point {(x, y)} is outside the {width}x{height} frame")

    writer = None
    if save_annotated_video:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create annotated output: {output_path}")

    analyzer = TrafficAnalyzer(
        camera_id=camera_id,
        zones=zones,
        stationary_speed_px_s=stationary_speed_px_s,
        pedestrian_clearance_hold_s=pedestrian_clearance_hold_s,
        waiting_threshold_s=waiting_threshold_s,
        blockage_threshold_s=blockage_threshold_s,
        blockage_medium_s=blockage_medium_s,
        blockage_high_s=blockage_high_s,
    )
    frame_index = 0
    inference_frame_count = 0
    inference_seconds = 0.0
    frame_started = time.perf_counter()
    visible_vehicle_counts: list[int] = []
    vehicle_box_area_percentages: list[float] = []
    # Appearance summaries stay in memory only for cross-camera association.
    # Crops, plates, and descriptors are never persisted.
    tracklets: dict[int, dict[str, Any]] = {}
    last_descriptor_time: dict[int, float] = {}
    first_preview_sent = False
    last_preview_at = 0.0
    last_metrics_at = 0.0
    last_video_time = 0.0
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            media_position_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            fallback_position_s = frame_index / fps
            video_time = media_position_s if math.isfinite(media_position_s) and media_position_s > last_video_time else fallback_position_s
            video_time = max(last_video_time, video_time)
            last_video_time = video_time
            final_frame = frame_count > 0 and frame_index == frame_count - 1
            if frame_index % frame_stride != 0 and not final_frame:
                frame_index += 1
                if progress_callback and (frame_index % 5 == 0 or frame_index == frame_count):
                    progress_callback(frame_index, frame_count)
                continue
            inference_started = time.perf_counter()
            detections = detector.detect_and_track(frame)
            inference_seconds += time.perf_counter() - inference_started
            inference_frame_count += 1
            labels = analyzer.update(detections, video_time)
            confirmed_vehicle_ids = {
                track_id for track_id, state in analyzer.states.items()
                if state.confirmed and state.category in VEHICLE_CLASSES
            }
            vehicle_detections = [
                detection
                for detection in detections
                if detection.category.lower() in VEHICLE_CLASSES
                and detection.track_id is not None
                and analyzer.track_aliases.get(detection.track_id, detection.track_id) in confirmed_vehicle_ids
                and (
                    camera_id != "approach"
                    or any(zone.kind == "approach" and zone.contains(detection.footpoint) for zone in zones)
                )
            ]
            vehicle_boxes = [detection.box for detection in vehicle_detections]
            visible_vehicle_counts.append(len(vehicle_detections))
            frame_area = width * height
            vehicle_box_area = sum(
                max(0.0, x2 - x1) * max(0.0, y2 - y1)
                for x1, y1, x2, y2 in vehicle_boxes
            )
            vehicle_box_area_percentages.append(vehicle_box_area / frame_area * 100)
            for detection in detections:
                if detection.track_id is None or detection.category.lower() not in VEHICLE_CLASSES:
                    continue
                if camera_id == "approach" and not any(
                    zone.kind == "approach" and zone.contains(detection.footpoint) for zone in zones
                ):
                    continue
                if video_time - last_descriptor_time.get(detection.track_id, -1e9) < 0.5:
                    continue
                last_descriptor_time[detection.track_id] = video_time
                x1, y1, x2, y2 = (int(v) for v in detection.box)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue
                crop = frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
                feature = hist.flatten().astype(float).tolist()
                item = tracklets.get(detection.track_id)
                if item is None:
                    tracklets[detection.track_id] = {
                        "track_id": detection.track_id, "category": detection.category.lower(),
                        "start_s": video_time + identity_time_offset_s, "end_s": video_time + identity_time_offset_s, "samples": 1,
                        "color_hist": feature,
                    }
                else:
                    n = item["samples"]
                    item["color_hist"] = [(old * n + new) / (n + 1) for old, new in zip(item["color_hist"], feature)]
                    item["samples"] = n + 1
                    item["end_s"] = video_time + identity_time_offset_s
            if writer is not None:
                annotated = _draw_overlay(cv2, frame.copy(), zones, detections, labels, frame_index, frame_count)
                writer.write(annotated)
            frame_index += 1
            now = time.perf_counter()
            urgent_pedestrian_event = (
                analyzer.metrics.pedestrian_in_free_left_route_now
                or analyzer.metrics.pedestrian_vehicle_overlap_now
                or analyzer.metrics.turn_path_blocked_now
            )
            if metrics_callback and (
                now - last_metrics_at >= 0.2
                or inference_frame_count == 1
                or urgent_pedestrian_event
            ):
                provisional_metrics = analyzer.metrics.to_dict()
                open_blockage_durations = [max(0.0, video_time - started) for started in analyzer.open_blockages.values()]
                provisional_metrics.update({
                    "frames_processed": inference_frame_count,
                    "total_observation_seconds": round(max(analyzer.metrics.total_observation_seconds, video_time), 2),
                    "total_stopped_delay_seconds": round(sum(state.stopped_delay_s for state in analyzer.states.values()), 2),
                    "total_blocked_seconds": round(analyzer.metrics.blocked_seconds + sum(open_blockage_durations), 2),
                    "maximum_blockage_duration_seconds": round(max(
                        [float(event.get("duration_seconds", 0)) for event in analyzer.metrics.blockage_events]
                        + open_blockage_durations + [0.0]
                    ), 2),
                })
                metrics_callback(provisional_metrics)
                last_metrics_at = now
            if preview_callback and (not first_preview_sent or now - last_preview_at >= 1 / TARGET_FPS or frame_index == frame_count):
                visible_pedestrians = sum(
                    detection.category.lower() == "person" and not is_rider_detection(detection, detections)
                    for detection in detections
                )
                vehicles_by_zone = {
                    zone.name: sum(
                        detection.category.lower() in VEHICLE_CLASSES and zone.contains(detection.footpoint)
                        for detection in detections
                    )
                    for zone in zones
                }
                live_annotated = _draw_overlay(cv2, frame.copy(), zones, detections, labels, frame_index, frame_count)
                elapsed = max(now - frame_started, 0.001)
                counts = {
                    "vehicles_in_frame": len(vehicle_detections),
                    "free_left_vehicles_in_frame": sum(
                        any(zone.kind == "free_left" and zone.contains(detection.footpoint) for zone in zones)
                        for detection in vehicle_detections
                    ) if camera_id == "approach" else 0,
                    "pedestrians_in_frame": visible_pedestrians,
                    "turn_path_objects_in_frame": analyzer.metrics.turn_path_object_count_now,
                    "turn_path_stationary_vehicle_count": analyzer.metrics.turn_path_stationary_vehicle_count_now,
                    "pedestrians_in_free_left_route": analyzer.metrics.pedestrian_in_free_left_route_count_now,
                    "vehicle_tracks": analyzer.metrics.relevant_vehicle_track_count if camera_id in {"approach", "exit"} else sum(analyzer.metrics.category_counts.values()),
                    "approach_lane_vehicle_tracks": analyzer.metrics.approach_lane_vehicle_track_count,
                    "free_left_vehicle_tracks": analyzer.metrics.relevant_vehicle_track_count if camera_id == "approach" else 0,
                    "free_left_queue_length_pixels": analyzer.metrics.current_queue_extent_pixels,
                    "maximum_free_left_queue_length_pixels": analyzer.metrics.maximum_queue_extent_pixels,
                    "free_left_stopped_delay_seconds": round(sum(state.stopped_delay_s for state in analyzer.states.values()), 2) if camera_id == "approach" else 0,
                    "inference_frames": inference_frame_count,
                    "pedestrian_tracks": analyzer.metrics.pedestrian_tracks,
                    "vehicles_by_zone_in_frame": vehicles_by_zone,
                    "vehicle_tracks_by_zone": dict(analyzer.metrics.zone_vehicle_tracks),
                    "processing_fps": round(frame_index / elapsed, 2),
                    "inference_fps": round(inference_frame_count / max(inference_seconds, 0.001), 2),
                    "realtime_factor": round(video_time / elapsed, 2),
                    "source_position_seconds": round(video_time, 2),
                    "wall_elapsed_seconds": round(elapsed, 2),
                    "source_fps": round(fps, 2),
                    "inference_stride": frame_stride,
                    "joining_road_moving_vehicles": analyzer.metrics.joining_flow_last_moving_vehicles,
                    "joining_road_stationary_vehicles": analyzer.metrics.joining_flow_last_stationary_vehicles,
                    "joining_road_unclassified_vehicles": analyzer.metrics.joining_flow_last_unclassified_vehicles,
                    "joining_road_vehicles": analyzer.metrics.joining_flow_last_vehicles,
                }
                duration_label = f"{frame_count / fps:.1f}s" if frame_count > 0 else "unknown"
                live_annotated = _draw_performance_hud(cv2, live_annotated, [
                    f"ANALYSIS {counts['inference_fps']:.1f} FPS / TARGET {TARGET_FPS}  |  PROCESSED {counts['processing_fps']:.1f} FPS",
                    f"CLIP {video_time:.1f}s / {duration_label}  |  STRIDE 1/{frame_stride}  |  {counts['realtime_factor']:.2f}x real time",
                ])
                preview_callback(live_annotated, counts)
                first_preview_sent = True
                last_preview_at = now
            if progress_callback and (frame_index % 5 == 0 or frame_index == frame_count):
                progress_callback(frame_index, frame_count)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_index == 0 and stop_event is not None and stop_event.is_set():
        input_warnings.append("Analysis was stopped before this clip produced a decodable frame.")
    elif frame_index == 0:
        raise RuntimeError(f"Video opened but no decodable frames were read: {source_path}")
    if frame_count > frame_index and not (stop_event is not None and stop_event.is_set()):
        input_warnings.append(f"Video decoding ended after {frame_index} of {frame_count} reported frames.")
    elif frame_count > 0 and frame_index > frame_count:
        input_warnings.append(f"The file reported {frame_count} frames, but {frame_index} frames were decoded; its frame-count metadata is inaccurate.")

    elapsed = time.perf_counter() - frame_started
    clip_duration = max(last_video_time + 1 / fps, frame_count / fps if frame_count > frame_index else 0.0)
    metrics = analyzer.finish(clip_duration)
    metrics.update(
        {
            "camera_id": camera_id,
            "source_file": source_path.name,
            "annotated_file": output_path.name if save_annotated_video else None,
            "input_fps": round(fps, 3),
            "video_decoder": capture_backend,
            "input_warnings": input_warnings,
            "frame_count_expected": frame_count,
            "frame_count_processed": frame_index,
            "inference_frame_count": inference_frame_count,
            "inference_stride": frame_stride,
            "frame_coverage_percent": round(min(100.0, frame_index / frame_count * 100), 2) if frame_count > 0 else None,
            "clip_duration_seconds": round(clip_duration, 2),
            "wall_time_seconds": round(elapsed, 2),
            "inference_fps": round(inference_frame_count / inference_seconds, 2) if inference_seconds > 0 else None,
            "end_to_end_fps": round(frame_index / elapsed, 2) if elapsed > 0 else None,
            "realtime_factor": round(clip_duration / elapsed, 2) if elapsed > 0 else None,
            "calibration": "not_configured",
            "average_visible_vehicles_per_frame": round(sum(visible_vehicle_counts) / len(visible_vehicle_counts), 2) if visible_vehicle_counts else 0,
            "maximum_visible_vehicles_in_frame": max(visible_vehicle_counts, default=0),
            "average_vehicle_bounding_box_area_percent": round(sum(vehicle_box_area_percentages) / len(vehicle_box_area_percentages), 2) if vehicle_box_area_percentages else 0,
            "maximum_vehicle_bounding_box_area_percent": round(max(vehicle_box_area_percentages, default=0), 2),
            "bounding_box_area_note": "Approximate summed vehicle bounding-box area; overlaps can be counted more than once and this is not calibrated physical road occupancy.",
            "stopped_early": bool(stop_event and stop_event.is_set()),
        }
    )
    metrics["_vehicle_tracklets"] = list(tracklets.values())
    return metrics


def _draw_overlay(cv2: Any, frame: Any, zones: list[Zone], detections: list[Any], labels: list[str], frame_index: int, frame_count: int) -> Any:
    """Draw ROIs and compact detection labels without face/image persistence."""

    import numpy as np

    colors = {"free_left": (0, 0, 255), "approach": (0, 220, 255), "pedestrian_conflict": (255, 0, 255), "exit": (255, 160, 0)}
    for zone in zones:
        points = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        color = colors.get(zone.kind, (120, 230, 120))
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        display_name = {"queue_area": "QUEUE AREA", "free_left_lane": "FREE LEFT", "joining_road_flow": "JOINING ROAD FLOW"}.get(zone.name, zone.name)
        if zone.kind == "free_left":
            label_x = max(min(point[0] for point in zone.polygon), max(point[0] for point in zone.polygon) - 160)
            label_y = max(point[1] for point in zone.polygon) - 12
            label_origin = (label_x, label_y)
        else:
            label_origin = zone.polygon[0]
        cv2.putText(frame, display_name, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for detection, label in zip(detections, labels):
        x1, y1, x2, y2 = (int(value) for value in detection.box)
        is_pedestrian = detection.category.lower() == "person"
        color = (255, 220, 0) if is_pedestrian else (0, 255, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_color = (0, 0, 255) if "BLOCKAGE" in label else color
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
    cv2.putText(frame, f"frame {frame_index + 1}/{frame_count or '?'}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def _draw_performance_hud(cv2: Any, frame: Any, lines: list[str]) -> Any:
    """Add readable performance telemetry after the frame is anonymized."""

    if not lines:
        return frame
    height, width = frame.shape[:2]
    if width < 64 or height < 64:
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(0.62, width / 1600))
    line_height = max(19, int(28 * scale / 0.55))
    x, y = 12, 44
    box_width = min(width - 16, max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 18)
    box_height = line_height * len(lines) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - 21), (x + box_width, min(height - 4, y - 21 + box_height)), (5, 17, 18), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line[:120], (x + 8, y + index * line_height), font, scale, (180, 255, 220), 1, cv2.LINE_AA)
    return frame
