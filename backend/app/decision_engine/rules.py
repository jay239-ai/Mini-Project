"""Explainable, quality-gated traffic advisory rules.

The output ranks situations for human review. It never controls a signal or
claims a calibrated phase duration. Historical reports can add context, but
cannot independently raise the live/recorded severity.
"""

from dataclasses import dataclass
import math
import time
from typing import Any


CAMERA_ROLES = ("approach", "turning", "exit")
REQUIRED_ZONE_KINDS = {
    "approach": {"approach", "free_left"},
    "turning": set(),
    "exit": set(),
}
REQUIRED_FIELDS = (
    "frames_processed", "total_observation_seconds", "maximum_queue_vehicles",
    "p95_queue_vehicles", "total_stopped_delay_seconds", "total_blocked_seconds",
    "maximum_blockage_duration_seconds",
    "pedestrian_vehicle_overlap_samples", "roi_configuration_valid",
    "relevant_vehicle_track_count", "turn_path_obstruction_count_now", "turn_path_blocked_now",
    "approach_lane_vehicle_track_count", "p95_queue_length_ratio",
)


@dataclass(frozen=True)
class DecisionThresholds:
    """Initial site thresholds; tune only after reviewing calibrated data."""

    queue_medium: int = 5
    queue_high: int = 10
    blockage_medium_seconds: float = 10.0
    blockage_high_seconds: float = 30.0
    stopped_delay_medium_vehicle_seconds: float = 30.0
    stopped_delay_high_vehicle_seconds: float = 60.0
    score_medium: int = 35
    score_high: int = 65
    joining_flow_stationary_ratio_medium: float = 0.65
    queue_length_medium_ratio: float = 0.35
    queue_length_high_ratio: float = 0.65
    minimum_observation_seconds: float = 2.0
    minimum_confidence_frames_per_camera: int = 30
    pedestrian_clearance_hold_seconds: float = 2.0

    def __post_init__(self) -> None:
        if min(self.queue_medium, self.queue_high) < 1 or self.queue_high <= self.queue_medium:
            raise ValueError("Queue thresholds must be positive and the high threshold must exceed medium.")
        if min(self.blockage_medium_seconds, self.blockage_high_seconds, self.stopped_delay_medium_vehicle_seconds, self.stopped_delay_high_vehicle_seconds) <= 0:
            raise ValueError("Time thresholds must be positive.")
        if self.blockage_high_seconds <= self.blockage_medium_seconds or self.stopped_delay_high_vehicle_seconds <= self.stopped_delay_medium_vehicle_seconds:
            raise ValueError("High time thresholds must exceed medium thresholds.")
        if not 1 <= self.score_medium < self.score_high <= 100:
            raise ValueError("Risk score thresholds must be increasing values between 1 and 100.")
        if not 0.5 <= self.joining_flow_stationary_ratio_medium <= 1.0:
            raise ValueError("Joining-road stationary ratio must be between 0.5 and 1.0.")
        if not 0.0 < self.queue_length_medium_ratio < self.queue_length_high_ratio <= 1.0:
            raise ValueError("Queue-length ratios must increase and remain in (0, 1].")
        if self.minimum_observation_seconds <= 0 or self.minimum_confidence_frames_per_camera < 1 or self.pedestrian_clearance_hold_seconds < 0:
            raise ValueError("Observation, frame-count, and pedestrian-clearance thresholds must be valid positive values.")


def valid_zone_configuration(camera_id: str, zone_rows: list[dict[str, Any]]) -> bool:
    """Only camera 1 needs a calibrated ROI; views 2 and 3 use the full frame."""

    expected = REQUIRED_ZONE_KINDS.get(camera_id)
    if expected is None or not isinstance(zone_rows, list):
        return False
    kinds = {str(row.get("kind", "")) for row in zone_rows if isinstance(row, dict)}
    return expected.issubset(kinds)


def _number(value: Any, name: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be a whole number")
    return int(value) if integer else float(value)


def _empty(severity: str, reason: str, *, camera_count: int, warnings: list[str]) -> dict[str, Any]:
    return {
        "severity": severity,
        "recommended_action": "collect_more_camera_data",
        "reason": reason,
        "triggered_rule": "insufficient_or_invalid_evidence",
        "risk_score": None,
        "confidence": "low",
        "decision_status": "insufficient_data",
        "advisory_only": True,
        "traffic_light": {
            "color": "yellow",
            "label": "YELLOW · INSUFFICIENT DATA",
            "evidence_quality": "Limited evidence. The system needs complete, current camera observations before showing green.",
        },
        "input_metrics": {"camera_count": camera_count},
        "data_quality_warnings": warnings,
        "historical_context": {"used_for_current_severity": False},
    }


def recommend(
    camera_metrics: list[dict[str, Any]],
    thresholds: DecisionThresholds | None = None,
    historical_context: dict[str, Any] | None = None,
    identity_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank current traffic evidence, refusing incomplete, stale or invalid data.

    Current score uses the worst camera-local queue, stopped delay, blockage,
    and conflict-zone evidence; it never sums repeated camera vehicle counts.
    """

    rules = thresholds or DecisionThresholds()
    if not isinstance(camera_metrics, list):
        return _empty("insufficient", "Camera measurements are malformed.", camera_count=0, warnings=["camera_metrics must be a list"])

    # Camera 2 is aimed at the conflict path. Any recognized object in its full
    # frame is a hard stop; this does not require a polygon or another camera.
    for item in camera_metrics:
        if not isinstance(item, dict) or item.get("camera_id") not in CAMERA_ROLES:
            continue
        role = item["camera_id"]
        if role == "approach" and item.get("roi_configuration_valid") is not True:
            continue
        if item.get("source_mode") == "live":
            observed_at = item.get("updated_at_epoch")
            if not isinstance(observed_at, (int, float)) or not 0 <= time.time() - float(observed_at) <= 2.5:
                continue
        try:
            conflict_samples = _number(item.get("pedestrian_vehicle_overlap_samples", 0), "pedestrian_vehicle_overlap_samples", integer=True)
        except ValueError:
            continue
        # Camera 1 covers the approach, not the actual crossing path. People
        # anywhere in that wide view must not hold the free-left signal.
        route_pedestrian = False
        active_overlap = role == "turning" and item.get("pedestrian_vehicle_overlap_now") is True
        path_obstructed = role == "turning" and item.get("turn_path_blocked_now") is True
        if path_obstructed or route_pedestrian or active_overlap:
            turn_path_count = int(item.get("turn_path_obstruction_count_now", 0) or 0)
            route_count = int(item.get("pedestrian_in_free_left_route_count_now", 0) or 0)
            if path_obstructed:
                reason = (
                    f"The turn path was recently obstructed; its {rules.pedestrian_clearance_hold_seconds:g}-second clearance interval is still active."
                    if turn_path_count == 0 else
                    f"The turn path currently has a pedestrian or a vehicle stopped long enough to block it. Wait for {rules.pedestrian_clearance_hold_seconds:g} seconds after it clears."
                )
                red_label = "RED · CLEARANCE HOLD" if turn_path_count == 0 else "RED · TURN PATH BLOCKED"
            elif route_pedestrian:
                reason = (
                    f"A person was recently detected in camera 1's free-left path; waiting for the {rules.pedestrian_clearance_hold_seconds:g}-second clearance interval."
                    if route_count == 0 else
                    "A person is currently detected in camera 1's free-left turn path."
                )
                red_label = "RED · CLEARANCE HOLD" if route_count == 0 else "RED · PERSON IN FREE-LEFT PATH"
            else:
                reason = f"A pedestrian and vehicle were recently together in a configured conflict area. Hold the turn for the {rules.pedestrian_clearance_hold_seconds:g}-second clearance interval."
                red_label = "RED · PEDESTRIAN CONFLICT"
            urgent = _empty("high", reason, camera_count=len(camera_metrics), warnings=[])
            urgent.update(
                recommended_action="hold_free_left_for_pedestrian_clearance",
                triggered_rule="blocked_turn_path" if path_obstructed else "person_in_free_left_path" if route_pedestrian else "pedestrian_vehicle_conflict",
                decision_status="operator_review_required",
                traffic_light={
                    "color": "red",
                    "label": red_label,
                    "evidence_quality": "A current obstruction is detected." if turn_path_count or route_count or active_overlap else "The last obstruction is no longer counted in the current frame; the short clearance hold remains active.",
                },
                input_metrics={
                    "pedestrian_in_free_left_route_now": route_pedestrian,
                    "pedestrians_in_free_left_route_now": route_count,
                    "pedestrian_vehicle_overlap_now": active_overlap,
                    "pedestrian_vehicle_overlap_samples": int(conflict_samples),
                    "turn_path_obstruction_count_now": turn_path_count,
                    "pedestrian_clearance_hold_seconds": rules.pedestrian_clearance_hold_seconds,
                    "camera_id": item["camera_id"],
                },
            )
            return urgent
    if len(camera_metrics) != 3:
        return _empty("insufficient", "All three camera roles are required for a junction advisory.", camera_count=len(camera_metrics), warnings=["exactly three camera summaries are required"])

    by_role: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for item in camera_metrics:
        if not isinstance(item, dict):
            warnings.append("A camera summary is not an object")
            continue
        role = item.get("camera_id")
        if role not in CAMERA_ROLES or role in by_role:
            warnings.append("Camera roles are missing, unknown, or duplicated")
            continue
        by_role[role] = item
    if len(by_role) != 3:
        return _empty("insufficient", "Camera roles are incomplete or duplicated.", camera_count=len(by_role), warnings=warnings or ["approach, turning, and exit are required"])

    # Camera 2 uses its calibrated polygon when available, otherwise it safely
    # falls back to checking the entire frame. Other cameras cannot block.
    turning_view = by_role["turning"]
    pedestrian_route_now = turning_view.get("pedestrian_in_free_left_route_now") is True
    pedestrian_route_count_now = int(turning_view.get("pedestrian_in_free_left_route_count_now", 0) or 0)
    pedestrian_vehicle_overlap_now = turning_view.get("pedestrian_vehicle_overlap_now") is True

    normalized: dict[str, dict[str, float | int]] = {}
    for role in CAMERA_ROLES:
        item = by_role[role]
        missing = [key for key in REQUIRED_FIELDS if key not in item]
        if missing:
            warnings.append(f"{role} is missing fields: {', '.join(missing)}")
            continue
        if item.get("roi_configuration_valid") is not True:
            warnings.append(f"{role} ROI configuration does not cover its required movement/conflict zones")
        try:
            values = {
                "frames_processed": _number(item["frames_processed"], f"{role}.frames_processed", integer=True),
                "total_observation_seconds": _number(item["total_observation_seconds"], f"{role}.total_observation_seconds"),
                "maximum_queue_vehicles": _number(item["maximum_queue_vehicles"], f"{role}.maximum_queue_vehicles", integer=True),
                "queue_vehicles_now": _number(item.get("queue_vehicles_now", 0), f"{role}.queue_vehicles_now", integer=True),
                "queue_length_ratio_now": _number(item.get("queue_length_ratio_now", 0), f"{role}.queue_length_ratio_now"),
                "p95_queue_vehicles": _number(item["p95_queue_vehicles"], f"{role}.p95_queue_vehicles", integer=True),
                "p95_queue_length_ratio": _number(item["p95_queue_length_ratio"], f"{role}.p95_queue_length_ratio"),
                "total_stopped_delay_seconds": _number(item["total_stopped_delay_seconds"], f"{role}.total_stopped_delay_seconds"),
                "total_blocked_seconds": _number(item["total_blocked_seconds"], f"{role}.total_blocked_seconds"),
                "maximum_blockage_duration_seconds": _number(item["maximum_blockage_duration_seconds"], f"{role}.maximum_blockage_duration_seconds"),
                "pedestrian_vehicle_overlap_samples": _number(item["pedestrian_vehicle_overlap_samples"], f"{role}.pedestrian_vehicle_overlap_samples", integer=True),
                "relevant_vehicle_track_count": _number(item["relevant_vehicle_track_count"], f"{role}.relevant_vehicle_track_count", integer=True),
                "approach_lane_vehicle_track_count": _number(item["approach_lane_vehicle_track_count"], f"{role}.approach_lane_vehicle_track_count", integer=True),
                "turn_path_obstruction_count_now": _number(item["turn_path_obstruction_count_now"], f"{role}.turn_path_obstruction_count_now", integer=True),
            }
            if values["p95_queue_length_ratio"] > 1:
                raise ValueError(f"{role}.p95_queue_length_ratio must not exceed 1")
            if values["queue_length_ratio_now"] > 1:
                raise ValueError(f"{role}.queue_length_ratio_now must not exceed 1")
            categories = item.get("relevant_vehicle_category_tracks", {})
            if not isinstance(categories, dict):
                raise ValueError(f"{role}.relevant_vehicle_category_tracks must be an object")
            category_total = sum(int(_number(value, f"{role}.relevant_vehicle_category_tracks", integer=True)) for value in categories.values())
            values["vehicle_track_count"] = values["relevant_vehicle_track_count"]
            if category_total != values["vehicle_track_count"]:
                raise ValueError(f"{role} relevant vehicle total does not match its category counts")
            if role == "exit":
                for field in (
                    "joining_flow_frames", "joining_flow_occupied_frames", "joining_flow_vehicle_samples",
                    "joining_flow_moving_samples", "joining_flow_stationary_samples",
                    "joining_flow_unclassified_samples", "joining_flow_max_stationary_vehicles",
                    "joining_flow_max_moving_vehicles",
                ):
                    values[field] = _number(item.get(field), f"exit.{field}", integer=True)
                values["joining_flow_max_stationary_duration_seconds"] = _number(
                    item.get("joining_flow_max_stationary_duration_seconds"),
                    "exit.joining_flow_max_stationary_duration_seconds",
                )
                values["joining_flow_last_moving_vehicles"] = _number(item.get("joining_flow_last_moving_vehicles", 0), "exit.joining_flow_last_moving_vehicles", integer=True)
                values["joining_flow_last_stationary_vehicles"] = _number(item.get("joining_flow_last_stationary_vehicles", 0), "exit.joining_flow_last_stationary_vehicles", integer=True)
                values["joining_flow_last_unclassified_vehicles"] = _number(item.get("joining_flow_last_unclassified_vehicles", 0), "exit.joining_flow_last_unclassified_vehicles", integer=True)
                values["joining_flow_last_vehicles"] = _number(item.get("joining_flow_last_vehicles", 0), "exit.joining_flow_last_vehicles", integer=True)
                if values["joining_flow_frames"] < 2 or values["joining_flow_occupied_frames"] > values["joining_flow_frames"]:
                    raise ValueError("exit joining-road flow frame measurements are inconsistent")
                classified = values["joining_flow_moving_samples"] + values["joining_flow_stationary_samples"] + values["joining_flow_unclassified_samples"]
                if classified > values["joining_flow_vehicle_samples"]:
                    raise ValueError("exit joining-road movement samples exceed detected vehicle samples")
            if values["frames_processed"] < 2 or values["total_observation_seconds"] <= 0:
                warnings.append(f"{role} has too little observation time")
            if item.get("source_mode") == "live" and (not isinstance(item.get("updated_at_epoch"), (int, float)) or item["updated_at_epoch"] <= 0):
                warnings.append(f"{role} live measurements have no freshness timestamp")
            elif item.get("source_mode") == "live" and time.time() - float(item["updated_at_epoch"]) > 2.5:
                warnings.append(f"{role} live measurements are stale (older than 2.5 seconds)")
            normalized[role] = values
        except (ValueError, TypeError) as error:
            warnings.append(str(error))

    if warnings or len(normalized) != 3:
        return _empty("insufficient", "The advisory is withheld because camera data is incomplete, stale, invalid, or uncalibrated.", camera_count=len(normalized), warnings=warnings or ["not all camera summaries passed validation"])

    max_queue = max(int(values["maximum_queue_vehicles"]) for values in normalized.values())
    p95_queue = max(int(values["p95_queue_vehicles"]) for values in normalized.values())
    max_blockage = max(float(values["maximum_blockage_duration_seconds"]) for values in normalized.values())
    approach_values = normalized["approach"]
    p95_queue_length_ratio = float(approach_values["p95_queue_length_ratio"])
    max_delay = float(approach_values["total_stopped_delay_seconds"]) / max(int(approach_values["vehicle_track_count"]), 1)
    conflict_samples = sum(int(values["pedestrian_vehicle_overlap_samples"]) for values in normalized.values())
    joining = normalized["exit"]
    joining_vehicle_samples = int(joining["joining_flow_vehicle_samples"])
    joining_moving_samples = int(joining["joining_flow_moving_samples"])
    joining_stationary_samples = int(joining["joining_flow_stationary_samples"])
    joining_unclassified_samples = int(joining["joining_flow_unclassified_samples"])
    joining_known_samples = joining_moving_samples + joining_stationary_samples
    if joining["joining_flow_last_vehicles"] and joining["joining_flow_last_unclassified_vehicles"] > joining["joining_flow_last_moving_vehicles"] + joining["joining_flow_last_stationary_vehicles"]:
        return _empty("insufficient", "The joining-road camera sees current vehicles but cannot classify whether they are moving; wait for clearer tracking before showing green.", camera_count=3, warnings=["current joining-road movement classification is not reliable"])
    joining_stationary_ratio = joining_stationary_samples / joining_known_samples if joining_known_samples else 0.0
    joining_max_stopped = int(joining["joining_flow_max_stationary_vehicles"])
    joining_max_moving = int(joining["joining_flow_max_moving_vehicles"])
    joining_max_stopped_seconds = float(joining["joining_flow_max_stationary_duration_seconds"])
    joining_now_moving = int(joining["joining_flow_last_moving_vehicles"])
    joining_now_stationary = int(joining["joining_flow_last_stationary_vehicles"])
    joining_now_count = int(joining["joining_flow_last_vehicles"])
    joining_flow_clear = joining_now_count == 0
    joining_flow_alert = joining_now_stationary > 0 and joining_now_moving == 0
    approach_now_queue = int(approach_values["queue_vehicles_now"])
    approach_now_length = float(approach_values["queue_length_ratio_now"])
    approach_pressure = approach_now_queue >= rules.queue_medium or approach_now_length >= rules.queue_length_medium_ratio

    short_observation_roles = [role for role, values in normalized.items() if float(values["total_observation_seconds"]) < rules.minimum_observation_seconds]
    if short_observation_roles and not conflict_samples:
        return _empty("insufficient", f"Collecting the first {rules.minimum_observation_seconds:g} seconds of camera evidence before assessing traffic pressure.", camera_count=3, warnings=[f"too little observation time: {', '.join(short_observation_roles)}"])

    queue_count_points = min(30.0, approach_now_queue / max(rules.queue_high, 1) * 30)
    queue_length_points = min(10.0, approach_now_length / max(rules.queue_length_high_ratio, 0.01) * 10)
    queue_points = round(queue_count_points + queue_length_points, 1)
    # Delay and blockage fields are accumulated over the whole clip, so they
    # are useful report context but cannot be presented as current risk points.
    delay_points = 0.0
    blockage_points = 0.0
    conflict_points = 25 if pedestrian_vehicle_overlap_now else 0
    current_score = min(100, round(queue_points + delay_points + blockage_points + conflict_points))

    # Historical reports remain visible as context, but the light uses only
    # current camera evidence so an old report cannot change the requested rule.
    history = historical_context if isinstance(historical_context, dict) else {}
    nested_history = history.get("decision_context")
    if isinstance(nested_history, dict):
        history = nested_history
    history_used = bool(history.get("matched_site") and history.get("report_count"))
    history_points = 0
    combined_score = current_score

    conflict = pedestrian_vehicle_overlap_now
    # Make the junction decision from simultaneous current-frame evidence.
    # Historical maxima and people elsewhere in camera 1 cannot hold the turn.
    joining_high = joining_now_moving >= rules.queue_medium
    high_rule = approach_pressure and joining_high
    joining_medium = approach_pressure and joining_now_count > 0
    medium_rule = joining_medium
    if conflict or high_rule:
        severity = "high"
        action = "urgent_operator_review_pedestrian_vehicle_conflict" if conflict else "prioritize_joining_road_blockage_review" if joining_high else "hold_for_turn_path_clearance"
        rule_name = "pedestrian_vehicle_conflict" if conflict else "high_joining_road_blockage" if joining_high else "blocked_turn_path"
        reason = (
            "The camera 2 turn-path view is occupied; hold the turn until it is clear for the configured clearance interval."
            if any(item.get("turn_path_blocked_now") for item in camera_metrics if isinstance(item, dict) and item.get("camera_id") == "turning")
            else "Current pedestrian conflict evidence requires holding the turn."
            if pedestrian_vehicle_overlap_now else "Both the free-left path and the joining road have high traffic moving toward the merge."
            if joining_high else "The camera 2 turn path is obstructed; wait for it to clear."
        )
    elif medium_rule or joining_medium:
        severity = "medium"
        action = "prioritize_joining_road_flow_review" if joining_medium else "prioritize_queue_and_delay_review"
        rule_name = "slow_or_stationary_joining_road_flow" if joining_medium else "medium_current_traffic_pressure"
        reason = "Camera 1 has a busy free-left approach while camera 3 currently has traffic using the merge. Wait for that competing flow to ease."
    else:
        severity, action, rule_name = "low", "continue_monitoring", "below_current_pressure_thresholds"
        reason = "Camera 2's turn path is clear and camera 3 has no competing flow, so camera 1 traffic alone does not prevent the free-left turn."

    historical_watch = False

    total_frames = sum(int(values["frames_processed"]) for values in normalized.values())
    min_observation = min(float(values["total_observation_seconds"]) for values in normalized.values())
    minimum_camera_frames = min(int(values["frames_processed"]) for values in normalized.values())
    confidence = "moderate" if minimum_camera_frames >= rules.minimum_confidence_frames_per_camera and min_observation >= rules.minimum_observation_seconds else "low"
    light_color = "red" if severity == "high" else "yellow" if severity == "medium" or confidence == "low" else "green"
    light_label = {
        "red": "RED · TURN PATH OCCUPIED" if conflict else "RED · FREE-LEFT AND JOINING FLOW HIGH",
        "yellow": "YELLOW · JOINING ROAD ACTIVE" if joining_medium else "YELLOW · FREE-LEFT FLOW CAUTION",
        "green": "GREEN · FREE-LEFT AVAILABLE",
    }[light_color]
    evidence_quality = f"At least {rules.minimum_observation_seconds:g}s and {rules.minimum_confidence_frames_per_camera} processed frames per view; camera 1's main-lane and free-left polygons must be calibrated. Camera 2 checks {'the configured turn-path polygon' if turning_view.get('turn_path_zone_configured') is True else 'its full frame'}; camera 3 uses its full frame." if confidence == "moderate" else "Limited evidence. Continue observing; green is withheld until all three views provide enough tracked frames."
    return {
        "severity": severity,
        "recommended_action": action,
        "reason": reason,
        "triggered_rule": rule_name,
        "risk_score": current_score,
        "combined_context_score": combined_score,
        "confidence": confidence,
        "traffic_light": {
            "color": light_color,
            "label": light_label,
            "evidence_quality": evidence_quality,
        },
        "decision_status": "operator_review_required" if severity == "high" else "advisory",
        "advisory_only": True,
        "input_metrics": {
            "maximum_camera_queue_vehicles": max_queue,
            "p95_camera_queue_vehicles": p95_queue,
            "camera_1_free_left_confirmed_tracks": int(approach_values["vehicle_track_count"]),
            "camera_1_main_lane_confirmed_tracks": int(approach_values["approach_lane_vehicle_track_count"]),
            "camera_1_free_left_current_max_occupancy": int(approach_values["maximum_queue_vehicles"]),
            "camera_1_free_left_queue_vehicles_now": approach_now_queue,
            "camera_1_free_left_queue_length_ratio_now": round(approach_now_length, 3),
            "camera_1_free_left_queue_length_ratio_p95": round(p95_queue_length_ratio, 3),
            "maximum_camera_blocked_seconds": round(max_blockage, 2),
            "maximum_camera_stopped_delay_vehicle_seconds": round(max_delay, 2),
            "total_camera_local_vehicle_tracks": sum(int(values["vehicle_track_count"]) for values in normalized.values()),
            "pedestrian_vehicle_overlap_samples": conflict_samples,
            "pedestrian_in_free_left_route_now": pedestrian_route_now,
            "pedestrians_in_free_left_route_now": pedestrian_route_count_now,
            "pedestrian_vehicle_overlap_now": pedestrian_vehicle_overlap_now,
            "camera_2_pedestrian_scope": "turn_path_polygon" if turning_view.get("turn_path_zone_configured") is True else "full_frame",
            "pedestrian_clearance_hold_seconds": rules.pedestrian_clearance_hold_seconds,
            "joining_road_flow": {
                "status": "clear" if joining_flow_clear else "stationary_or_slow" if joining_flow_alert else "moving" if joining_now_moving else "uncertain",
                "moving_vehicle_observations": joining_moving_samples,
                "stationary_vehicle_observations": joining_stationary_samples,
                "stationary_observation_ratio": round(joining_stationary_ratio, 3),
                "maximum_moving_vehicles_in_frame": joining_max_moving,
                "moving_vehicles_now": joining_now_moving,
                "stationary_vehicles_now": joining_now_stationary,
                "vehicles_now": joining_now_count,
                "maximum_stationary_vehicles": joining_max_stopped,
                "longest_stationary_duration_seconds": round(joining_max_stopped_seconds, 2),
                "unclassified_observations": joining_unclassified_samples,
            },
            "current_evidence_score_components": {
                "queue_count_points": round(queue_count_points, 1),
                "queue_length_points": round(queue_length_points, 1),
                "stopped_delay_points": delay_points,
                "blockage_points": blockage_points,
                "conflict_points": conflict_points,
            },
        },
        "data_quality_warnings": [],
        "historical_context": {
            "used_for_current_severity": historical_watch,
            "matched_site": bool(history.get("matched_site")),
            "eligible_for_context": history_used,
            "report_count": int(history.get("report_count", 0) or 0),
            "peak_congestion_level": history.get("peak_congestion_level"),
            "maximum_queue_vehicles": history.get("maximum_queue_vehicles"),
            "maximum_queue_length_meters": history.get("maximum_queue_length_meters"),
            "report_period_end": history.get("report_period_end"),
            "context_points": history_points,
            "note": "Historical reports are shown for context only and never change the current signal color." if history_used else "No historical report matched this site; only current camera evidence was used.",
        },
        "decision_limits": [
            "Not a traffic signal command or a safe phase duration.",
            "ROI calibration, detector misses, and camera synchronization affect these estimates.",
            "Moderate is the maximum confidence before site-specific validation against reviewed ground truth.",
        ],
    }
