"""Zone-based counts, queue/wait measures, blockage, and pedestrian indicators."""

from collections import Counter
from dataclasses import dataclass, field

from vision_engine.detection.yolo import Detection
from vision_engine.roi.zones import Zone

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
RIDER_SUPPORT_CLASSES = {"motorcycle", "bicycle"}


@dataclass
class TrackState:
    """Short-lived history required to compute movement and time measures."""

    category: str
    first_seen: float
    last_seen: float
    last_point: tuple[int, int]
    last_box: tuple[float, float, float, float]
    last_moving_at: float
    last_speed_px_s: float = 0.0
    zones: set[str] = field(default_factory=set)
    maximum_stationary_interval_s: float = 0.0
    stopped_delay_s: float = 0.0
    movement_direction: str = "unknown"
    observation_count: int = 1
    category_votes: Counter[str] = field(default_factory=Counter)
    confirmed: bool = False


@dataclass
class TrafficMetrics:
    """Session summary for one camera, with no cross-camera deduplication."""

    frames_processed: int = 0
    unique_tracks: int = 0
    category_counts: Counter[str] = field(default_factory=Counter)
    relevant_vehicle_category_tracks: Counter[str] = field(default_factory=Counter)
    relevant_vehicle_track_count: int = 0
    approach_lane_vehicle_category_tracks: Counter[str] = field(default_factory=Counter)
    approach_lane_vehicle_track_count: int = 0
    zone_vehicle_tracks: Counter[str] = field(default_factory=Counter)
    pedestrian_tracks: int = 0
    zone_entries: Counter[str] = field(default_factory=Counter)
    zone_exits: Counter[str] = field(default_factory=Counter)
    maximum_queue: int = 0
    queue_samples: list[int] = field(default_factory=list)
    queue_length_ratio_samples: list[float] = field(default_factory=list)
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
    turn_path_obstruction_count_now: int = 0
    turn_path_blocked_now: bool = False
    turn_path_pedestrian_count_now: int = 0
    turn_path_pedestrian_now: bool = False
    turn_path_stationary_vehicle_count_now: int = 0
    turn_path_object_count_now: int = 0
    turn_path_obstruction_frames: int = 0
    turn_path_max_obstruction_count: int = 0
    joining_flow_frames: int = 0
    joining_flow_occupied_frames: int = 0
    joining_flow_vehicle_samples: int = 0
    joining_flow_moving_samples: int = 0
    joining_flow_stationary_samples: int = 0
    joining_flow_unclassified_samples: int = 0
    joining_flow_max_stationary_vehicles: int = 0
    joining_flow_max_moving_vehicles: int = 0
    joining_flow_max_stationary_duration_seconds: float = 0.0
    joining_flow_last_moving_vehicles: int = 0
    joining_flow_last_stationary_vehicles: int = 0
    joining_flow_last_unclassified_vehicles: int = 0
    joining_flow_last_vehicles: int = 0
    total_observation_seconds: float = 0.0
    blocked_seconds: float = 0.0
    total_tracker_seconds: float = 0.0
    total_stopped_delay_seconds: float = 0.0
    current_queue_extent_pixels: float = 0.0
    maximum_queue_extent_pixels: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Convert counters and samples into JSON-friendly summary values."""

        waits = self.waiting_samples_seconds
        duration = self.total_observation_seconds
        ordered_queues = sorted(self.queue_samples)
        p95_index = max(0, min(len(ordered_queues) - 1, int(0.95 * (len(ordered_queues) - 1)))) if ordered_queues else None
        ordered_queue_ratios = sorted(self.queue_length_ratio_samples)
        p95_ratio_index = max(0, min(len(ordered_queue_ratios) - 1, int(0.95 * (len(ordered_queue_ratios) - 1)))) if ordered_queue_ratios else None
        return {
            "frames_processed": self.frames_processed,
            "unique_tracks_camera_local": self.unique_tracks,
            "vehicle_category_tracks": dict(self.category_counts),
            "relevant_vehicle_category_tracks": dict(self.relevant_vehicle_category_tracks),
            "relevant_vehicle_track_count": self.relevant_vehicle_track_count,
            "approach_lane_vehicle_category_tracks": dict(self.approach_lane_vehicle_category_tracks),
            "approach_lane_vehicle_track_count": self.approach_lane_vehicle_track_count,
            "zone_vehicle_tracks": dict(self.zone_vehicle_tracks),
            "pedestrian_tracks": self.pedestrian_tracks,
            "zone_entries": dict(self.zone_entries),
            "zone_exits": dict(self.zone_exits),
            "maximum_queue_vehicles": self.maximum_queue,
            "average_queue_vehicles": round(sum(self.queue_samples) / len(self.queue_samples), 2) if self.queue_samples else None,
            "p95_queue_vehicles": ordered_queues[p95_index] if p95_index is not None else None,
            "p95_queue_length_ratio": round(ordered_queue_ratios[p95_ratio_index], 3) if p95_ratio_index is not None else 0.0,
            "average_moving_approach_vehicles": round(sum(self.moving_approach_samples) / len(self.moving_approach_samples), 2) if self.moving_approach_samples else None,
            "maximum_moving_approach_vehicles": max(self.moving_approach_samples, default=0),
            "average_stationary_approach_vehicles": round(sum(self.stationary_approach_samples) / len(self.stationary_approach_samples), 2) if self.stationary_approach_samples else None,
            "maximum_stationary_approach_vehicles": max(self.stationary_approach_samples, default=0),
            "average_wait_seconds": round(sum(waits) / len(waits), 2) if waits else None,
            "maximum_wait_seconds": round(max(waits), 2) if waits else None,
            "total_stopped_delay_seconds": round(self.total_stopped_delay_seconds, 2),
            "current_queue_extent_pixels": round(self.current_queue_extent_pixels, 1),
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
            "turn_path_obstruction_count_now": self.turn_path_obstruction_count_now,
            "turn_path_blocked_now": self.turn_path_blocked_now,
            "turn_path_pedestrian_count_now": self.turn_path_pedestrian_count_now,
            "turn_path_pedestrian_now": self.turn_path_pedestrian_now,
            "turn_path_stationary_vehicle_count_now": self.turn_path_stationary_vehicle_count_now,
            "turn_path_object_count_now": self.turn_path_object_count_now,
            "turn_path_obstruction_frames": self.turn_path_obstruction_frames,
            "turn_path_max_obstruction_count": self.turn_path_max_obstruction_count,
            "joining_flow_frames": self.joining_flow_frames,
            "joining_flow_occupied_frames": self.joining_flow_occupied_frames,
            "joining_flow_vehicle_samples": self.joining_flow_vehicle_samples,
            "joining_flow_moving_samples": self.joining_flow_moving_samples,
            "joining_flow_stationary_samples": self.joining_flow_stationary_samples,
            "joining_flow_unclassified_samples": self.joining_flow_unclassified_samples,
            "joining_flow_max_stationary_vehicles": self.joining_flow_max_stationary_vehicles,
            "joining_flow_max_moving_vehicles": self.joining_flow_max_moving_vehicles,
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
        self.track_aliases: dict[int, int] = {}
        self.metrics = TrafficMetrics()
        self.open_blockages: dict[int, float] = {}
        self.counted_tracks: set[int] = set()
        self.relevant_track_categories: dict[int, str] = {}
        self.approach_lane_track_categories: dict[int, str] = {}
        self.counted_zone_vehicle_tracks: set[tuple[int, str]] = set()
        self.last_turn_path_obstruction_s: float | None = None
        self.last_turn_path_pedestrian_s: float | None = None
        self.previous_frame_time: float | None = None
        free_left_points = [point for zone in zones if zone.kind == "free_left" for point in zone.polygon]
        if len(free_left_points) >= 3:
            xs = [point[0] for point in free_left_points]
            ys = [point[1] for point in free_left_points]
            self.free_left_roi_diagonal = max(((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5, 1.0)
        else:
            self.free_left_roi_diagonal = 0.0

    def update(self, detections: list[Detection], timestamp_s: float) -> list[str]:
        """Update confirmed object tracks, path obstruction, and camera-specific flow."""

        self.metrics.frames_processed += 1
        if self.previous_frame_time is not None:
            self.metrics.total_observation_seconds += max(0.0, timestamp_s - self.previous_frame_time)
        self.previous_frame_time = timestamp_s
        visible_states: set[int] = set()
        moving_approach = stationary_approach = 0
        approach_points: list[tuple[int, int]] = []
        joining_flow_vehicles = joining_flow_stationary_vehicles = 0
        joining_flow_moving_vehicles = joining_flow_unclassified_vehicles = 0
        pedestrian_in_free_left_route = False
        pedestrians_in_free_left_route = 0
        annotations: list[str] = []
        current_track_ids = {detection.track_id for detection in detections if detection.track_id is not None}
        rider_detection_ids = {
            id(person) for person in detections
            if person.category.lower() == "person" and is_rider_detection(person, detections)
        }
        turn_path_pedestrians = sum(
            detection.category.lower() == "person" and id(detection) not in rider_detection_ids
            for detection in detections
        ) if self.camera_id == "turning" else 0
        turn_path_stationary_vehicles = 0
        if self.camera_id == "turning" and turn_path_pedestrians:
            self.last_turn_path_pedestrian_s = timestamp_s

        full_frame_joining = self.camera_id == "exit" and not self.zones
        has_joining_flow = full_frame_joining or bool(self.joining_flow_zone_names)

        for detection in detections:
            point = detection.footpoint
            object_zones = {zone.name for zone in self.zones if zone.contains(point)}
            raw_category = detection.category.lower()
            if id(detection) in rider_detection_ids:
                annotations.append("rider on bicycle/motorcycle · not a pedestrian")
                continue
            if raw_category == "person" and any(self.zone_by_name[name].kind == "free_left" for name in object_zones):
                pedestrian_in_free_left_route = True
                pedestrians_in_free_left_route += 1
            if detection.track_id is None:
                if raw_category in VEHICLE_CLASSES and has_joining_flow:
                    self.metrics.joining_flow_vehicle_samples += 1
                    self.metrics.joining_flow_unclassified_samples += 1
                    joining_flow_vehicles += 1
                    joining_flow_unclassified_vehicles += 1
                annotations.append(f"{detection.category} (untracked)")
                continue

            track_id = self.track_aliases.get(detection.track_id, detection.track_id)
            if track_id not in self.states:
                merged_id = self._nearby_fragment(detection, timestamp_s, current_track_ids)
                if merged_id is not None:
                    self.track_aliases[detection.track_id] = merged_id
                    track_id = merged_id
            visible_states.add(track_id)
            state = self.states.get(track_id)
            had_previous_observation = state is not None and timestamp_s > state.last_seen
            elapsed = 0.0
            if state is None:
                state = TrackState(
                    raw_category, timestamp_s, timestamp_s, point, detection.box, timestamp_s,
                    zones=object_zones,
                    category_votes=Counter({raw_category: max(float(detection.confidence), 0.01)}),
                )
                self.states[track_id] = state
                previous_zones: set[str] = set()
            else:
                elapsed = max(0.0, timestamp_s - state.last_seen)
                if elapsed > 0:
                    dx = point[0] - state.last_point[0]
                    dy = point[1] - state.last_point[1]
                    state.last_speed_px_s = (dx * dx + dy * dy) ** 0.5 / elapsed
                    if state.last_speed_px_s > 0:
                        state.movement_direction = _movement_direction(dx, dy)
                        self.metrics.relative_speed_px_s_samples.append(state.last_speed_px_s)
                        self.metrics.movement_direction_samples[state.movement_direction] += 1
                previous_zones = set(state.zones)
                state.last_point = point
                state.last_box = detection.box
                state.last_seen = timestamp_s
                state.observation_count += 1
                state.category_votes[raw_category] += max(float(detection.confidence), 0.01)
                if state.last_speed_px_s >= self._motion_threshold(state.last_box):
                    state.last_moving_at = timestamp_s

            was_confirmed = state.confirmed
            previous_category = state.category
            state.category = max(state.category_votes, key=state.category_votes.get)
            if not state.confirmed and state.observation_count >= 2:
                state.confirmed = True
                self.metrics.unique_tracks += 1
                self._adjust_category_count(state.category, 1)
            elif was_confirmed and state.category != previous_category:
                self._adjust_category_count(previous_category, -1)
                self._adjust_category_count(state.category, 1)

            category = state.category
            is_vehicle = category in VEHICLE_CLASSES
            is_joining_flow = is_vehicle and (full_frame_joining or any(self.zone_by_name[name].kind == "exit" for name in object_zones))
            in_free_left = any(self.zone_by_name[name].kind == "free_left" for name in object_zones)
            in_main_approach = any(self.zone_by_name[name].kind == "approach" for name in object_zones)
            relevant_vehicle = is_vehicle and (
                (self.camera_id == "approach" and in_free_left)
                or full_frame_joining
            )

            if state.confirmed:
                if category == "person" and not was_confirmed:
                    self.metrics.pedestrian_tracks += 1
                entered_zones = object_zones if not was_confirmed else object_zones - previous_zones
                for zone_name in entered_zones:
                    self.metrics.zone_entries[zone_name] += 1
                if was_confirmed:
                    for zone_name in previous_zones - object_zones:
                        self.metrics.zone_exits[zone_name] += 1
                if is_vehicle:
                    for zone_name in object_zones:
                        zone_track_key = (track_id, zone_name)
                        if zone_track_key not in self.counted_zone_vehicle_tracks:
                            self.counted_zone_vehicle_tracks.add(zone_track_key)
                            self.metrics.zone_vehicle_tracks[zone_name] += 1
                if self.camera_id == "approach" and is_vehicle and in_main_approach:
                    prior_lane_category = self.approach_lane_track_categories.get(track_id)
                    if prior_lane_category != category:
                        if prior_lane_category:
                            self.metrics.approach_lane_vehicle_category_tracks[prior_lane_category] -= 1
                        self.approach_lane_track_categories[track_id] = category
                        self.metrics.approach_lane_vehicle_category_tracks[category] += 1
                elif self.camera_id == "approach" and not is_vehicle and track_id in self.approach_lane_track_categories:
                    prior_lane_category = self.approach_lane_track_categories.pop(track_id)
                    self.metrics.approach_lane_vehicle_category_tracks[prior_lane_category] -= 1
                self.metrics.approach_lane_vehicle_track_count = len(self.approach_lane_track_categories)
                if relevant_vehicle:
                    old_category = self.relevant_track_categories.get(track_id)
                    if old_category != category:
                        if old_category:
                            self.metrics.relevant_vehicle_category_tracks[old_category] -= 1
                        self.relevant_track_categories[track_id] = category
                        self.metrics.relevant_vehicle_category_tracks[category] += 1
                elif not is_vehicle and track_id in self.relevant_track_categories:
                    old_category = self.relevant_track_categories.pop(track_id)
                    self.metrics.relevant_vehicle_category_tracks[old_category] -= 1
                self.metrics.relevant_vehicle_track_count = len(self.relevant_track_categories)

            state.zones = object_zones
            occupied_free_left = self.camera_id == "approach" and in_free_left
            is_approach = self.camera_id == "approach" and occupied_free_left
            stationary_for = max(0.0, timestamp_s - state.last_moving_at)

            if is_joining_flow:
                if state.confirmed:
                    self.metrics.joining_flow_vehicle_samples += 1
                    joining_flow_vehicles += 1
                    if had_previous_observation:
                        if state.last_speed_px_s >= self._motion_threshold(state.last_box):
                            self.metrics.joining_flow_moving_samples += 1
                            joining_flow_moving_vehicles += 1
                        else:
                            self.metrics.joining_flow_stationary_samples += 1
                            joining_flow_stationary_vehicles += 1
                            state.stopped_delay_s += elapsed
                    else:
                        self.metrics.joining_flow_unclassified_samples += 1
                        joining_flow_unclassified_vehicles += 1
                    self.metrics.joining_flow_max_stationary_duration_seconds = max(
                        self.metrics.joining_flow_max_stationary_duration_seconds, stationary_for
                    )
            if is_vehicle and is_approach and state.confirmed:
                approach_points.append(point)
                if state.last_speed_px_s < self._motion_threshold(state.last_box):
                    stationary_approach += 1
                    state.stopped_delay_s += elapsed
                else:
                    moving_approach += 1
            if is_vehicle and stationary_for >= self.waiting_threshold_s and (is_approach or is_joining_flow):
                state.maximum_stationary_interval_s = max(state.maximum_stationary_interval_s, stationary_for)
            if is_vehicle and occupied_free_left and stationary_for >= self.blockage_threshold_s:
                self.open_blockages.setdefault(track_id, timestamp_s - stationary_for)
            elif track_id in self.open_blockages:
                started = self.open_blockages.pop(track_id)
                duration = max(0.0, timestamp_s - started)
                self.metrics.blocked_seconds += duration
                self.metrics.blockage_events.append({"camera_id": self.camera_id, "track_id": track_id, "started_at_video_seconds": round(started, 3), "ended_at_video_seconds": round(timestamp_s, 3), "duration_seconds": round(duration, 3), "severity": self._severity(duration)})
            annotation = f"{category} #{track_id}" if state.confirmed else f"{category} #{track_id} confirming"
            if occupied_free_left and stationary_for >= self.blockage_threshold_s:
                annotation += f" BLOCKAGE {stationary_for:.1f}s"
            annotations.append(annotation)

            if self.camera_id == "turning" and is_vehicle and state.confirmed and stationary_for >= self.blockage_threshold_s:
                turn_path_stationary_vehicles += 1

        turn_path_blocked_vehicles = turn_path_stationary_vehicles
        turn_path_obstruction_count = turn_path_pedestrians + turn_path_blocked_vehicles
        if self.camera_id == "turning" and turn_path_obstruction_count:
            self.last_turn_path_obstruction_s = timestamp_s
            self.metrics.turn_path_obstruction_frames += 1
            self.metrics.turn_path_max_obstruction_count = max(
                self.metrics.turn_path_max_obstruction_count, turn_path_obstruction_count
            )
        self.metrics.turn_path_object_count_now = len(detections) if self.camera_id == "turning" else 0
        self.metrics.turn_path_pedestrian_count_now = turn_path_pedestrians
        self.metrics.turn_path_pedestrian_now = bool(
            self.last_turn_path_pedestrian_s is not None
            and timestamp_s - self.last_turn_path_pedestrian_s <= self.pedestrian_clearance_hold_s
        )
        self.metrics.turn_path_stationary_vehicle_count_now = turn_path_stationary_vehicles
        self.metrics.turn_path_obstruction_count_now = turn_path_obstruction_count
        self.metrics.turn_path_blocked_now = bool(
            self.last_turn_path_obstruction_s is not None
            and timestamp_s - self.last_turn_path_obstruction_s <= self.pedestrian_clearance_hold_s
        )
        if pedestrian_in_free_left_route:
            self.metrics.pedestrian_in_free_left_samples += 1
            self.last_pedestrian_route_seen_s = timestamp_s
        self.metrics.pedestrian_in_free_left_route_count_now = pedestrians_in_free_left_route
        self.metrics.pedestrian_in_free_left_route_now = bool(
            self.last_pedestrian_route_seen_s is not None
            and timestamp_s - self.last_pedestrian_route_seen_s <= self.pedestrian_clearance_hold_s
        )
        if has_joining_flow:
            self.metrics.joining_flow_frames += 1
            if joining_flow_vehicles:
                self.metrics.joining_flow_occupied_frames += 1
            self.metrics.joining_flow_max_stationary_vehicles = max(self.metrics.joining_flow_max_stationary_vehicles, joining_flow_stationary_vehicles)
            self.metrics.joining_flow_max_moving_vehicles = max(self.metrics.joining_flow_max_moving_vehicles, joining_flow_moving_vehicles)
            self.metrics.joining_flow_last_moving_vehicles = joining_flow_moving_vehicles
            self.metrics.joining_flow_last_stationary_vehicles = joining_flow_stationary_vehicles
            self.metrics.joining_flow_last_unclassified_vehicles = joining_flow_unclassified_vehicles
            self.metrics.joining_flow_last_vehicles = joining_flow_vehicles
        queue_size = moving_approach + stationary_approach if self.camera_id == "approach" else 0
        self.metrics.moving_approach_samples.append(moving_approach if self.camera_id == "approach" else 0)
        self.metrics.stationary_approach_samples.append(stationary_approach if self.camera_id == "approach" else 0)
        self.metrics.queue_samples.append(queue_size)
        self.metrics.maximum_queue = max(self.metrics.maximum_queue, queue_size)
        self.metrics.current_queue_extent_pixels = 0.0
        if len(approach_points) > 1:
            xs = [point[0] for point in approach_points]
            ys = [point[1] for point in approach_points]
            extent = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            self.metrics.current_queue_extent_pixels = extent
            self.metrics.maximum_queue_extent_pixels = max(self.metrics.maximum_queue_extent_pixels, extent)
        queue_length_ratio = min(1.0, self.metrics.current_queue_extent_pixels / self.free_left_roi_diagonal) if self.free_left_roi_diagonal else 0.0
        self.metrics.queue_length_ratio_samples.append(queue_length_ratio)
        for track_id in set(self.open_blockages) - visible_states:
            started = self.open_blockages.pop(track_id)
            duration = max(0.0, timestamp_s - started)
            self.metrics.blocked_seconds += duration
            self.metrics.blockage_events.append({"camera_id": self.camera_id, "track_id": track_id, "started_at_video_seconds": round(started, 3), "ended_at_video_seconds": round(timestamp_s, 3), "duration_seconds": round(duration, 3), "severity": self._severity(duration), "ended_reason": "track_not_visible"})
        return annotations

    def _adjust_category_count(self, category: str, delta: int) -> None:
        """Keep reported vehicle/person categories on confirmed consensus labels."""

        normalized = category.lower()
        if normalized in VEHICLE_CLASSES:
            self.metrics.category_counts[normalized] += delta
            if self.metrics.category_counts[normalized] <= 0:
                self.metrics.category_counts.pop(normalized, None)
        elif normalized == "person":
            self.metrics.pedestrian_tracks = max(0, self.metrics.pedestrian_tracks + delta)

    def _motion_threshold(self, box: tuple[float, float, float, float]) -> float:
        """Scale the image-motion threshold with object size to reduce perspective bias."""

        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        # A fixed 8 px/s floor classified small, distant vehicles as stopped
        # even while visibly moving. Keep a modest jitter floor and scale with
        # object size so slow approaches remain moving.
        return max(min(self.stationary_speed_px_s, 3.0), (width * width + height * height) ** 0.5 * 0.08)

    def _nearby_fragment(
        self,
        detection: Detection,
        timestamp_s: float,
        current_track_ids: set[int | None],
    ) -> int | None:
        """Join a short ByteTrack ID break when the same vehicle box continues."""

        if detection.category.lower() not in VEHICLE_CLASSES:
            return None
        x1, y1, x2, y2 = detection.box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        diagonal = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
        best_id: int | None = None
        best_score = 0.0
        for track_id, state in self.states.items():
            if track_id in current_track_ids or not state.confirmed or state.category not in VEHICLE_CLASSES:
                continue
            gap = timestamp_s - state.last_seen
            if gap < 0 or gap > 0.35:
                continue
            ax1, ay1, ax2, ay2 = state.last_box
            intersection_width = max(0.0, min(x2, ax2) - max(x1, ax1))
            intersection_height = max(0.0, min(y2, ay2) - max(y1, ay1))
            intersection = intersection_width * intersection_height
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            iou = intersection / max(area_a + area_b - intersection, 1.0)
            previous_center = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
            distance = ((center[0] - previous_center[0]) ** 2 + (center[1] - previous_center[1]) ** 2) ** 0.5
            proximity = max(0.0, 1.0 - distance / (diagonal * 0.35))
            score = max(iou, proximity * 0.6)
            if (iou >= 0.4 or distance <= diagonal * 0.22) and score > best_score:
                best_id, best_score = track_id, score
        return best_id

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


def is_rider_detection(person: Detection, detections: list[Detection]) -> bool:
    """Filter person boxes that visibly overlap a bicycle/motorcycle rider box."""

    px1, py1, px2, py2 = person.box
    person_width = max(px2 - px1, 1.0)
    person_center_x = (px1 + px2) / 2
    for vehicle in detections:
        if vehicle.category.lower() not in RIDER_SUPPORT_CLASSES:
            continue
        vx1, vy1, vx2, vy2 = vehicle.box
        vehicle_width = max(vx2 - vx1, 1.0)
        vehicle_height = max(vy2 - vy1, 1.0)
        horizontal_overlap = max(0.0, min(px2, vx2) - max(px1, vx1))
        center_aligned = vx1 - vehicle_width * 0.15 <= person_center_x <= vx2 + vehicle_width * 0.15
        lower_body_near_vehicle = py2 >= vy1 - vehicle_height * 0.2 and py2 <= vy2 + vehicle_height * 0.25
        if center_aligned and lower_body_near_vehicle and horizontal_overlap >= person_width * 0.2:
            return True
    return False
