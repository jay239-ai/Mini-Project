"""Zone-based counts, queue/wait measures, blockage, and pedestrian indicators."""

from collections import Counter
from dataclasses import dataclass, field

from vision_engine.detection.yolo import Detection
from vision_engine.roi.zones import Zone

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


@dataclass
class TrackState:
    """Short-lived history required to compute movement and time measures."""

    category: str
    first_seen: float
    last_seen: float
    last_point: tuple[int, int]
    last_moving_at: float
    last_speed_px_s: float = 0.0
    zones: set[str] = field(default_factory=set)
    maximum_stationary_interval_s: float = 0.0
    stopped_delay_s: float = 0.0
    movement_direction: str = "unknown"


@dataclass
class TrafficMetrics:
    """Session summary for one camera, with no cross-camera deduplication."""

    frames_processed: int = 0
    unique_tracks: int = 0
    category_counts: Counter[str] = field(default_factory=Counter)
    zone_vehicle_tracks: Counter[str] = field(default_factory=Counter)
    pedestrian_tracks: int = 0
    zone_entries: Counter[str] = field(default_factory=Counter)
    zone_exits: Counter[str] = field(default_factory=Counter)
    maximum_queue: int = 0
    queue_samples: list[int] = field(default_factory=list)
    moving_approach_samples: list[int] = field(default_factory=list)
    stationary_approach_samples: list[int] = field(default_factory=list)
    waiting_samples_seconds: list[float] = field(default_factory=list)
    relative_speed_px_s_samples: list[float] = field(default_factory=list)
    movement_direction_samples: Counter[str] = field(default_factory=Counter)
    blockage_events: list[dict[str, object]] = field(default_factory=list)
    pedestrian_conflict_samples: int = 0
    pedestrian_in_free_left_samples: int = 0
    pedestrian_in_free_left_route_now: bool = False
    pedestrian_in_free_left_route_count_now: int = 0
    pedestrian_vehicle_overlap_now: bool = False
    joining_flow_frames: int = 0
    joining_flow_occupied_frames: int = 0
    joining_flow_vehicle_samples: int = 0
    joining_flow_moving_samples: int = 0
    joining_flow_stationary_samples: int = 0
    joining_flow_unclassified_samples: int = 0
    joining_flow_max_stationary_vehicles: int = 0
    joining_flow_max_stationary_duration_seconds: float = 0.0
    joining_flow_last_moving_vehicles: int = 0
    joining_flow_last_stationary_vehicles: int = 0
    joining_flow_last_unclassified_vehicles: int = 0
    joining_flow_last_vehicles: int = 0
    total_observation_seconds: float = 0.0
    blocked_seconds: float = 0.0
    total_tracker_seconds: float = 0.0
    total_stopped_delay_seconds: float = 0.0
    maximum_queue_extent_pixels: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Convert counters and samples into JSON-friendly summary values."""

        waits = self.waiting_samples_seconds
        duration = self.total_observation_seconds
        ordered_queues = sorted(self.queue_samples)
        p95_index = max(0, min(len(ordered_queues) - 1, int(0.95 * (len(ordered_queues) - 1)))) if ordered_queues else None
        return {
            "frames_processed": self.frames_processed,
            "unique_tracks_camera_local": self.unique_tracks,
            "vehicle_category_tracks": dict(self.category_counts),
            "zone_vehicle_tracks": dict(self.zone_vehicle_tracks),
            "pedestrian_tracks": self.pedestrian_tracks,
            "zone_entries": dict(self.zone_entries),
            "zone_exits": dict(self.zone_exits),
            "maximum_queue_vehicles": self.maximum_queue,
            "average_queue_vehicles": round(sum(self.queue_samples) / len(self.queue_samples), 2) if self.queue_samples else None,
            "p95_queue_vehicles": ordered_queues[p95_index] if p95_index is not None else None,
            "average_moving_approach_vehicles": round(sum(self.moving_approach_samples) / len(self.moving_approach_samples), 2) if self.moving_approach_samples else None,
            "maximum_moving_approach_vehicles": max(self.moving_approach_samples, default=0),
            "average_stationary_approach_vehicles": round(sum(self.stationary_approach_samples) / len(self.stationary_approach_samples), 2) if self.stationary_approach_samples else None,
            "maximum_stationary_approach_vehicles": max(self.stationary_approach_samples, default=0),
            "average_wait_seconds": round(sum(waits) / len(waits), 2) if waits else None,
            "maximum_wait_seconds": round(max(waits), 2) if waits else None,
            "total_stopped_delay_seconds": round(self.total_stopped_delay_seconds, 2),
            "maximum_queue_extent_pixels": round(self.maximum_queue_extent_pixels, 1),
            "queue_extent_note": "Image-plane estimate only; meters require a calibrated road-plane homography and stop line.",
            "average_relative_speed_px_per_second": round(sum(self.relative_speed_px_s_samples) / len(self.relative_speed_px_s_samples), 2) if self.relative_speed_px_s_samples else None,
            "movement_direction_samples": dict(self.movement_direction_samples),
            "blockage_events": self.blockage_events,
            "total_blocked_seconds": round(self.blocked_seconds, 2),
            "maximum_blockage_duration_seconds": round(max((float(event.get("duration_seconds", 0)) for event in self.blockage_events), default=0.0), 2),
            "blockage_ratio_percent": round(self.blocked_seconds / duration * 100, 2) if duration > 0 else None,
            "pedestrian_vehicle_overlap_samples": self.pedestrian_conflict_samples,
            "pedestrian_in_free_left_samples": self.pedestrian_in_free_left_samples,
            "pedestrian_in_free_left_route_now": self.pedestrian_in_free_left_route_now,
            "pedestrian_in_free_left_route_count_now": self.pedestrian_in_free_left_route_count_now,
            "pedestrian_vehicle_overlap_now": self.pedestrian_vehicle_overlap_now,
            "joining_flow_frames": self.joining_flow_frames,
            "joining_flow_occupied_frames": self.joining_flow_occupied_frames,
            "joining_flow_vehicle_samples": self.joining_flow_vehicle_samples,
            "joining_flow_moving_samples": self.joining_flow_moving_samples,
            "joining_flow_stationary_samples": self.joining_flow_stationary_samples,
            "joining_flow_unclassified_samples": self.joining_flow_unclassified_samples,
            "joining_flow_max_stationary_vehicles": self.joining_flow_max_stationary_vehicles,
            "joining_flow_max_stationary_duration_seconds": round(self.joining_flow_max_stationary_duration_seconds, 2),
            "joining_flow_last_moving_vehicles": self.joining_flow_last_moving_vehicles,
            "joining_flow_last_stationary_vehicles": self.joining_flow_last_stationary_vehicles,
            "joining_flow_last_unclassified_vehicles": self.joining_flow_last_unclassified_vehicles,
            "joining_flow_last_vehicles": self.joining_flow_last_vehicles,
            "total_observation_seconds": round(duration, 2),
            "average_tracker_fps": round(self.frames_processed / self.total_tracker_seconds, 2) if self.total_tracker_seconds > 0 else None,
            "speed": {"value": None, "unit": None, "basis": "unavailable_without_camera_calibration"},
        }


class TrafficAnalyzer:
    """Consume detections in frame order and update camera-specific metrics."""

    def __init__(
        self,
        camera_id: str,
        zones: list[Zone],
        stationary_speed_px_s: float = 8.0,
        waiting_threshold_s: float = 10.0,
        blockage_threshold_s: float = 5.0,
        blockage_medium_s: float = 10.0,
        blockage_high_s: float = 30.0,
        pedestrian_clearance_hold_s: float = 2.0,
    ) -> None:
        if waiting_threshold_s < 0 or blockage_threshold_s < 0 or pedestrian_clearance_hold_s < 0:
            raise ValueError("Time thresholds cannot be negative")
        self.camera_id = camera_id
        self.zones = zones
        self.zone_by_name = {zone.name: zone for zone in zones}
        self.joining_flow_zone_names = {zone.name for zone in zones if zone.kind == "exit"}
        self.stationary_speed_px_s = stationary_speed_px_s
        self.waiting_threshold_s = waiting_threshold_s
        self.blockage_threshold_s = blockage_threshold_s
        if blockage_medium_s < blockage_threshold_s or blockage_high_s < blockage_medium_s:
            raise ValueError("Blockage severity thresholds must be increasing")
        self.blockage_medium_s = blockage_medium_s
        self.blockage_high_s = blockage_high_s
        self.pedestrian_clearance_hold_s = pedestrian_clearance_hold_s
        self.last_pedestrian_route_seen_s: float | None = None
        self.last_pedestrian_vehicle_conflict_s: float | None = None
        self.states: dict[int, TrackState] = {}
        self.metrics = TrafficMetrics()
        self.open_blockages: dict[int, float] = {}
        self.counted_tracks: set[int] = set()
        self.counted_zone_vehicle_tracks: set[tuple[int, str]] = set()
        self.previous_frame_time: float | None = None

    def update(self, detections: list[Detection], timestamp_s: float) -> list[str]:
        """Update counts and event indicators; return annotation labels."""

        self.metrics.frames_processed += 1
        if self.previous_frame_time is not None:
            self.metrics.total_observation_seconds += max(0.0, timestamp_s - self.previous_frame_time)
        self.previous_frame_time = timestamp_s
        visible_states: set[int] = set()
        pedestrian_in_conflict_zone = False
        vehicle_in_conflict_zone = False
        moving_approach = 0
        stationary_approach = 0
        approach_points: list[tuple[int, int]] = []
        joining_flow_vehicles = 0
        joining_flow_stationary_vehicles = 0
        joining_flow_moving_vehicles = 0
        joining_flow_unclassified_vehicles = 0
        pedestrian_in_free_left_route = False
        pedestrians_in_free_left_route = 0
        annotations: list[str] = []

        for detection in detections:
            point = detection.footpoint
            object_zones = {zone.name for zone in self.zones if zone.contains(point)}
            is_vehicle = detection.category.lower() in VEHICLE_CLASSES
            is_joining_flow = is_vehicle and any(self.zone_by_name[z].kind == "exit" for z in object_zones)
            if detection.category.lower() == "person" and any(self.zone_by_name[z].kind == "pedestrian_conflict" for z in object_zones):
                pedestrian_in_conflict_zone = True
            if detection.category.lower() == "person" and any(self.zone_by_name[z].kind in {"free_left", "pedestrian_conflict"} for z in object_zones):
                pedestrian_in_free_left_route = True
                pedestrians_in_free_left_route += 1
            # A vehicle in the tight crossing zone or its configured nearby
            # buffer is relevant only when a person is in the crossing zone.
            if is_vehicle and any(self.zone_by_name[z].kind in {"pedestrian_conflict", "pedestrian_nearby"} for z in object_zones):
                vehicle_in_conflict_zone = True
            if is_joining_flow:
                self.metrics.joining_flow_vehicle_samples += 1
                joining_flow_vehicles += 1

            if detection.track_id is None:
                if is_joining_flow:
                    self.metrics.joining_flow_unclassified_samples += 1
                    joining_flow_unclassified_vehicles += 1
                annotations.append(f"{detection.category} (untracked)")
                continue

            track_id = detection.track_id
            visible_states.add(track_id)
            state = self.states.get(track_id)
            had_previous_observation = state is not None and timestamp_s > state.last_seen
            elapsed = 0.0
            if state is None:
                state = TrackState(detection.category, timestamp_s, timestamp_s, point, timestamp_s, zones=object_zones)
                self.states[track_id] = state
                self.metrics.unique_tracks += 1
                if is_vehicle:
                    self.metrics.category_counts[detection.category] += 1
                elif detection.category.lower() == "person":
                    self.metrics.pedestrian_tracks += 1
                for zone_name in object_zones:
                    self.metrics.zone_entries[zone_name] += 1
                    if self.zone_by_name[zone_name].kind == "free_left":
                        self.counted_tracks.add(track_id)
            else:
                elapsed = timestamp_s - state.last_seen
                if elapsed > 0:
                    dx = point[0] - state.last_point[0]
                    dy = point[1] - state.last_point[1]
                    state.last_speed_px_s = (dx * dx + dy * dy) ** 0.5 / elapsed
                    if state.last_speed_px_s > 0:
                        state.movement_direction = _movement_direction(dx, dy)
                        self.metrics.relative_speed_px_s_samples.append(state.last_speed_px_s)
                        self.metrics.movement_direction_samples[state.movement_direction] += 1
                state.last_point = point
                state.last_seen = timestamp_s
                if state.last_speed_px_s >= self.stationary_speed_px_s:
                    state.last_moving_at = timestamp_s
                for zone_name in object_zones - state.zones:
                    self.metrics.zone_entries[zone_name] += 1
                    if zone_name in self.zone_by_name and self.zone_by_name[zone_name].kind == "free_left" and track_id not in self.counted_tracks:
                        self.counted_tracks.add(track_id)
                for zone_name in state.zones - object_zones:
                    self.metrics.zone_exits[zone_name] += 1
                state.zones = object_zones

            occupied_free_left = any(self.zone_by_name[z].kind == "free_left" for z in object_zones)
            stationary_for = max(0.0, timestamp_s - state.last_moving_at)
            is_approach = any(self.zone_by_name[z].kind == "approach" for z in object_zones)
            if is_joining_flow:
                if had_previous_observation:
                    if state.last_speed_px_s >= self.stationary_speed_px_s:
                        self.metrics.joining_flow_moving_samples += 1
                        joining_flow_moving_vehicles += 1
                    else:
                        self.metrics.joining_flow_stationary_samples += 1
                        joining_flow_stationary_vehicles += 1
                else:
                    self.metrics.joining_flow_unclassified_samples += 1
                    joining_flow_unclassified_vehicles += 1
            if is_vehicle and is_approach:
                approach_points.append(point)
                if state.last_speed_px_s < self.stationary_speed_px_s:
                    stationary_approach += 1
                    state.stopped_delay_s += max(0.0, elapsed)
                else:
                    moving_approach += 1
            elif is_joining_flow and state.last_speed_px_s < self.stationary_speed_px_s:
                state.stopped_delay_s += max(0.0, elapsed)
            if is_vehicle:
                for zone_name in object_zones:
                    zone_track_key = (track_id, zone_name)
                    if zone_track_key not in self.counted_zone_vehicle_tracks:
                        self.counted_zone_vehicle_tracks.add(zone_track_key)
                        self.metrics.zone_vehicle_tracks[zone_name] += 1
            if is_vehicle and stationary_for >= self.waiting_threshold_s:
                if object_zones.intersection({"approach", "free_left_lane"}) or any(self.zone_by_name[z].kind in {"approach", "free_left", "exit"} for z in object_zones):
                    state.maximum_stationary_interval_s = max(state.maximum_stationary_interval_s, stationary_for)
            if is_joining_flow:
                self.metrics.joining_flow_max_stationary_duration_seconds = max(self.metrics.joining_flow_max_stationary_duration_seconds, stationary_for)
            if is_vehicle and occupied_free_left and stationary_for >= self.blockage_threshold_s:
                self.open_blockages.setdefault(track_id, timestamp_s - stationary_for)
            elif track_id in self.open_blockages:
                started = self.open_blockages.pop(track_id)
                duration = max(0.0, timestamp_s - started)
                self.metrics.blocked_seconds += duration
                self.metrics.blockage_events.append({"camera_id": self.camera_id, "track_id": track_id, "started_at_video_seconds": round(started, 3), "ended_at_video_seconds": round(timestamp_s, 3), "duration_seconds": round(duration, 3), "severity": self._severity(duration)})
            annotation = f"{detection.category} #{track_id}"
            if occupied_free_left and stationary_for >= self.blockage_threshold_s:
                annotation += f" BLOCKAGE {stationary_for:.1f}s"
            annotations.append(annotation)

        if pedestrian_in_conflict_zone and vehicle_in_conflict_zone:
            self.metrics.pedestrian_conflict_samples += 1
            self.last_pedestrian_vehicle_conflict_s = timestamp_s
        if pedestrian_in_free_left_route:
            self.metrics.pedestrian_in_free_left_samples += 1
            self.last_pedestrian_route_seen_s = timestamp_s
        self.metrics.pedestrian_in_free_left_route_count_now = pedestrians_in_free_left_route
        self.metrics.pedestrian_in_free_left_route_now = bool(
            self.last_pedestrian_route_seen_s is not None
            and timestamp_s - self.last_pedestrian_route_seen_s <= self.pedestrian_clearance_hold_s
        )
        self.metrics.pedestrian_vehicle_overlap_now = bool(
            self.last_pedestrian_vehicle_conflict_s is not None
            and timestamp_s - self.last_pedestrian_vehicle_conflict_s <= self.pedestrian_clearance_hold_s
        )
        if self.joining_flow_zone_names:
            self.metrics.joining_flow_frames += 1
            if joining_flow_vehicles:
                self.metrics.joining_flow_occupied_frames += 1
            self.metrics.joining_flow_max_stationary_vehicles = max(self.metrics.joining_flow_max_stationary_vehicles, joining_flow_stationary_vehicles)
            self.metrics.joining_flow_last_moving_vehicles = joining_flow_moving_vehicles
            self.metrics.joining_flow_last_stationary_vehicles = joining_flow_stationary_vehicles
            self.metrics.joining_flow_last_unclassified_vehicles = joining_flow_unclassified_vehicles
            self.metrics.joining_flow_last_vehicles = joining_flow_vehicles
        queue_size = moving_approach + stationary_approach
        self.metrics.moving_approach_samples.append(moving_approach)
        self.metrics.stationary_approach_samples.append(stationary_approach)
        self.metrics.queue_samples.append(queue_size)
        self.metrics.maximum_queue = max(self.metrics.maximum_queue, queue_size)
        if len(approach_points) > 1:
            xs = [point[0] for point in approach_points]
            ys = [point[1] for point in approach_points]
            extent = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            self.metrics.maximum_queue_extent_pixels = max(self.metrics.maximum_queue_extent_pixels, extent)
        for track_id in set(self.open_blockages) - visible_states:
            started = self.open_blockages.pop(track_id)
            duration = max(0.0, timestamp_s - started)
            self.metrics.blocked_seconds += duration
            self.metrics.blockage_events.append({"camera_id": self.camera_id, "track_id": track_id, "started_at_video_seconds": round(started, 3), "ended_at_video_seconds": round(timestamp_s, 3), "duration_seconds": round(duration, 3), "severity": self._severity(duration), "ended_reason": "track_not_visible"})
        return annotations

    def finish(self, duration_s: float) -> dict[str, object]:
        """Close any open events at clip end and return a serializable summary."""

        for track_id, started in list(self.open_blockages.items()):
            duration = max(0.0, duration_s - started)
            self.metrics.blocked_seconds += duration
            self.metrics.blockage_events.append({"camera_id": self.camera_id, "track_id": track_id, "started_at_video_seconds": round(started, 3), "ended_at_video_seconds": round(duration_s, 3), "duration_seconds": round(duration, 3), "severity": self._severity(duration), "ended_reason": "clip_end"})
            self.open_blockages.pop(track_id, None)
        self.metrics.waiting_samples_seconds = [
            state.maximum_stationary_interval_s
            for state in self.states.values()
            if state.maximum_stationary_interval_s >= self.waiting_threshold_s
        ]
        self.metrics.total_stopped_delay_seconds = sum(state.stopped_delay_s for state in self.states.values())
        self.metrics.total_observation_seconds = max(self.metrics.total_observation_seconds, duration_s)
        return self.metrics.to_dict()

    def _severity(self, duration_s: float) -> str:
        """Assign severity using this camera's configured duration bands."""

        if duration_s >= self.blockage_high_s:
            return "high"
        if duration_s >= self.blockage_medium_s:
            return "medium"
        return "low"


def _movement_direction(dx: float, dy: float) -> str:
    """Classify image-plane motion; this is not a road/world direction."""

    horizontal = "right" if dx > 0 else "left"
    vertical = "down" if dy > 0 else "up"
    if abs(dx) > 1.5 * abs(dy):
        return horizontal
    if abs(dy) > 1.5 * abs(dx):
        return vertical
    return f"{vertical}_{horizontal}"
