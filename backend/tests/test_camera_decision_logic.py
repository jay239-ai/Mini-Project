"""Regression coverage for role-specific traffic rules and conservative counting."""

import unittest

from backend.app.decision_engine.rules import recommend, valid_zone_configuration
from vision_engine.detection.yolo import Detection
from vision_engine.metrics.traffic import TrafficAnalyzer
from vision_engine.roi.zones import Zone


def camera(role: str, **overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "camera_id": role,
        "frames_processed": 40,
        "total_observation_seconds": 4.0,
        "maximum_queue_vehicles": 0,
        "queue_vehicles_now": 0,
        "queue_length_ratio_now": 0.0,
        "p95_queue_vehicles": 0,
        "p95_queue_length_ratio": 0.0,
        "total_stopped_delay_seconds": 0.0,
        "total_blocked_seconds": 0.0,
        "maximum_blockage_duration_seconds": 0.0,
        "pedestrian_vehicle_overlap_samples": 0,
        "roi_configuration_valid": True,
        "relevant_vehicle_track_count": 0,
        "relevant_vehicle_category_tracks": {},
        "approach_lane_vehicle_track_count": 0,
        "turn_path_obstruction_count_now": 0,
        "turn_path_blocked_now": False,
        "turn_path_zone_configured": True,
        "joining_flow_frames": 40,
        "joining_flow_occupied_frames": 0,
        "joining_flow_vehicle_samples": 0,
        "joining_flow_moving_samples": 0,
        "joining_flow_stationary_samples": 0,
        "joining_flow_unclassified_samples": 0,
        "joining_flow_max_stationary_vehicles": 0,
        "joining_flow_max_moving_vehicles": 0,
        "joining_flow_max_stationary_duration_seconds": 0.0,
        "joining_flow_last_moving_vehicles": 0,
        "joining_flow_last_stationary_vehicles": 0,
        "joining_flow_last_unclassified_vehicles": 0,
        "joining_flow_last_vehicles": 0,
    }
    item.update(overrides)
    return item


class RoleSpecificDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [camera("approach"), camera("turning"), camera("exit")]

    def test_only_approach_requires_roi(self) -> None:
        self.assertFalse(valid_zone_configuration("approach", []))
        self.assertTrue(valid_zone_configuration("turning", []))
        self.assertTrue(valid_zone_configuration("exit", []))

    def test_blocked_turn_path_is_red_without_other_cameras(self) -> None:
        decision = recommend([camera("turning", turn_path_blocked_now=True, turn_path_obstruction_count_now=1)])
        self.assertEqual(decision["traffic_light"]["color"], "red")
        self.assertEqual(decision["triggered_rule"], "blocked_turn_path")

    def test_uncalibrated_camera_two_falls_back_to_full_frame(self) -> None:
        self.rows[1]["turn_path_zone_configured"] = False
        decision = recommend(self.rows)
        self.assertEqual(decision["traffic_light"]["color"], "green")
        self.assertEqual(decision["input_metrics"]["camera_2_pedestrian_scope"], "full_frame")

    def test_moving_vehicle_in_turn_path_does_not_force_red(self) -> None:
        self.rows[1]["turn_path_object_count_now"] = 1
        decision = recommend(self.rows)
        self.assertEqual(decision["traffic_light"]["color"], "green")

    def test_long_historical_queue_alone_does_not_hold_clear_joining_road(self) -> None:
        self.rows[0]["p95_queue_length_ratio"] = 0.7
        self.rows[0]["maximum_queue_vehicles"] = 12
        decision = recommend(self.rows)
        self.assertEqual(decision["traffic_light"]["color"], "green")

    def test_people_in_camera_one_do_not_block_free_left(self) -> None:
        self.rows[0]["pedestrian_in_free_left_route_now"] = True
        self.rows[0]["pedestrian_in_free_left_route_count_now"] = 1
        decision = recommend(self.rows)
        self.assertEqual(decision["traffic_light"]["color"], "green")

    def test_busy_free_left_and_moving_join_flow_is_red(self) -> None:
        self.rows[0].update(maximum_queue_vehicles=6, queue_vehicles_now=6, relevant_vehicle_track_count=6)
        self.rows[0]["relevant_vehicle_category_tracks"] = {"car": 6}
        self.rows[2].update(joining_flow_vehicle_samples=20, joining_flow_moving_samples=20,
                            joining_flow_max_moving_vehicles=5, joining_flow_last_moving_vehicles=5,
                            joining_flow_last_vehicles=5)
        decision = recommend(self.rows)
        self.assertEqual(decision["traffic_light"]["color"], "red")
        self.assertEqual(decision["triggered_rule"], "high_joining_road_blockage")

    def test_stationary_joining_traffic_alone_does_not_hold_clear_approach(self) -> None:
        self.rows[2].update(joining_flow_vehicle_samples=10, joining_flow_stationary_samples=10,
                            joining_flow_max_stationary_vehicles=4, joining_flow_last_stationary_vehicles=4,
                            joining_flow_last_vehicles=4)
        decision = recommend(self.rows)
        self.assertEqual(decision["traffic_light"]["color"], "green")

    def test_clear_observed_paths_can_be_green(self) -> None:
        decision = recommend(self.rows)
        self.assertEqual(decision["traffic_light"]["color"], "green")


class ConfirmedTrackCountingTests(unittest.TestCase):
    @staticmethod
    def turn_path_zone() -> Zone:
        return Zone("free_left_turn_path", ((0, 0), (100, 0), (100, 100), (0, 100)), "turn_path")

    def test_counts_once_after_confirmation_and_only_after_roi_entry(self) -> None:
        roi = Zone("free_left", ((0, 0), (100, 0), (100, 100), (0, 100)), "free_left")
        analyzer = TrafficAnalyzer("approach", [roi])
        outside = Detection(4, "car", 0.9, (150, 20, 180, 50))
        inside_bus = Detection(7, "bus", 0.45, (10, 10, 30, 40))
        inside_car = Detection(7, "car", 0.9, (10, 10, 30, 40))
        analyzer.update([outside, inside_bus], 0.0)
        self.assertEqual(analyzer.metrics.relevant_vehicle_track_count, 0)
        analyzer.update([outside, inside_car], 0.1)
        self.assertEqual(analyzer.metrics.relevant_vehicle_track_count, 1)
        self.assertEqual(dict(analyzer.metrics.relevant_vehicle_category_tracks), {"car": 1})
        analyzer.update([outside, inside_car], 0.2)
        self.assertEqual(analyzer.metrics.relevant_vehicle_track_count, 1)
        self.assertEqual(dict(analyzer.metrics.relevant_vehicle_category_tracks), {"car": 1})

    def test_full_frame_joining_camera_tracks_movement_without_roi(self) -> None:
        analyzer = TrafficAnalyzer("exit", [])
        analyzer.update([Detection(2, "car", 0.9, (10, 10, 30, 30))], 0.0)
        analyzer.update([Detection(2, "car", 0.9, (30, 10, 50, 30))], 1.0)
        self.assertEqual(analyzer.metrics.joining_flow_last_moving_vehicles, 1)
        self.assertEqual(analyzer.metrics.relevant_vehicle_track_count, 1)

    def test_short_tracker_id_break_does_not_create_duplicate_vehicle(self) -> None:
        analyzer = TrafficAnalyzer("exit", [])
        analyzer.update([Detection(2, "car", 0.9, (10, 10, 50, 50))], 0.0)
        analyzer.update([Detection(2, "car", 0.9, (12, 10, 52, 50))], 0.1)
        analyzer.update([Detection(9, "car", 0.9, (13, 10, 53, 50))], 0.2)
        self.assertEqual(analyzer.metrics.relevant_vehicle_track_count, 1)
        self.assertEqual(analyzer.metrics.unique_tracks, 1)
        self.assertEqual(analyzer.track_aliases[9], 2)

    def test_main_lane_and_free_left_counts_are_separate(self) -> None:
        main_lane = Zone("main_approach", ((0, 0), (100, 0), (100, 100), (0, 100)), "approach")
        free_left = Zone("free_left_lane", ((50, 0), (100, 0), (100, 100), (50, 100)), "free_left")
        analyzer = TrafficAnalyzer("approach", [main_lane, free_left])
        analyzer.update([
            Detection(1, "car", 0.9, (15, 10, 35, 40)),
            Detection(2, "car", 0.9, (65, 10, 85, 40)),
        ], 0.0)
        analyzer.update([
            Detection(1, "car", 0.9, (16, 10, 36, 40)),
            Detection(2, "car", 0.9, (66, 10, 86, 40)),
        ], 0.1)
        self.assertEqual(analyzer.metrics.approach_lane_vehicle_track_count, 2)
        self.assertEqual(analyzer.metrics.relevant_vehicle_track_count, 1)

    def test_person_box_on_motorcycle_is_not_counted_as_pedestrian(self) -> None:
        analyzer = TrafficAnalyzer("turning", [], blockage_threshold_s=2.0)
        rider = Detection(8, "person", 0.8, (20, 0, 50, 70))
        motorcycle = Detection(9, "motorcycle", 0.9, (15, 45, 55, 85))
        analyzer.update([rider, motorcycle], 0.0)
        analyzer.update([rider, motorcycle], 0.1)
        self.assertFalse(analyzer.metrics.turn_path_pedestrian_now)
        self.assertEqual(analyzer.metrics.pedestrian_tracks, 0)

    def test_person_in_turn_path_blocks_immediately(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()])
        analyzer.update([Detection(4, "person", 0.9, (10, 10, 30, 70))], 0.0)
        self.assertTrue(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_pedestrian_count_now, 1)

    def test_no_polygon_uses_full_frame_for_pedestrians(self) -> None:
        analyzer = TrafficAnalyzer("turning", [])
        analyzer.update([Detection(4, "person", 0.9, (110, 10, 130, 70))], 0.0)
        self.assertTrue(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_pedestrian_count_now, 1)

    def test_pedestrian_outside_turn_path_does_not_block(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()])
        analyzer.update([Detection(4, "person", 0.9, (110, 10, 130, 70))], 0.0)
        self.assertFalse(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_pedestrian_count_now, 0)

    def test_stationary_vehicle_outside_turn_path_does_not_block(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()], blockage_threshold_s=2.0)
        parked = Detection(3, "car", 0.9, (110, 10, 130, 50))
        for timestamp in (0.0, 0.1, 1.0, 2.1, 4.0):
            analyzer.update([parked], timestamp)
        self.assertFalse(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_stationary_vehicle_count_now, 0)

    def test_no_polygon_uses_full_frame_for_stationary_vehicle_blockage(self) -> None:
        analyzer = TrafficAnalyzer("turning", [], blockage_threshold_s=2.0)
        parked = Detection(3, "car", 0.9, (110, 10, 130, 50))
        for timestamp in (0.0, 0.1, 0.8, 1.5, 2.1):
            analyzer.update([parked], timestamp)
        self.assertTrue(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_stationary_vehicle_count_now, 1)

    def test_stationary_timer_starts_when_vehicle_enters_turn_path(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()], blockage_threshold_s=2.0)
        parked_outside = Detection(3, "car", 0.9, (90, 10, 110, 50))
        entered_path = Detection(3, "car", 0.9, (70, 10, 90, 50))
        for timestamp in (0.0, 1.0, 2.0, 4.0):
            analyzer.update([parked_outside], timestamp)
        analyzer.update([entered_path], 4.1)
        self.assertFalse(analyzer.metrics.turn_path_blocked_now)

    def test_stationary_timer_restarts_after_long_detection_gap(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()], blockage_threshold_s=2.0)
        parked = Detection(3, "car", 0.9, (10, 10, 50, 50))
        analyzer.update([parked], 0.0)
        analyzer.update([parked], 0.1)
        analyzer.update([parked], 5.0)
        self.assertFalse(analyzer.metrics.turn_path_blocked_now)
        for timestamp in (5.5, 6.0, 6.5, 7.0):
            analyzer.update([parked], timestamp)
        self.assertTrue(analyzer.metrics.turn_path_blocked_now)

    def test_stationary_vehicle_in_turn_path_blocks_after_threshold(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()], blockage_threshold_s=2.0)
        stationary = Detection(3, "car", 0.9, (10, 10, 50, 50))
        for timestamp in (0.0, 0.1, 0.8, 1.5, 2.1):
            analyzer.update([stationary], timestamp)
        self.assertTrue(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_stationary_vehicle_count_now, 1)

    def test_moving_turn_path_vehicle_does_not_become_stationary_blockage(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()], blockage_threshold_s=2.0)
        for second in range(7):
            car = Detection(3, "car", 0.9, (second * 20, 10, second * 20 + 40, 50))
            analyzer.update([car], float(second))
        self.assertFalse(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_stationary_vehicle_count_now, 0)

    def test_vehicle_approaching_camera_is_not_stationary_when_box_expands(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()], blockage_threshold_s=2.0)
        for frame in range(31):
            size = 30 + frame * 2
            car = Detection(3, "motorcycle", 0.9, (50 - size / 2, 100 - size, 50 + size / 2, 100))
            analyzer.update([car], frame * 0.1)
        self.assertFalse(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_stationary_vehicle_count_now, 0)

    def test_slow_small_turn_path_vehicle_is_not_misclassified_as_stationary(self) -> None:
        analyzer = TrafficAnalyzer("turning", [self.turn_path_zone()], blockage_threshold_s=2.0)
        for sample in range(13):
            x = sample * 3
            car = Detection(12, "car", 0.9, (x, 10, x + 12, 22))
            analyzer.update([car], sample * 0.5)
        self.assertFalse(analyzer.metrics.turn_path_blocked_now)
        self.assertEqual(analyzer.metrics.turn_path_stationary_vehicle_count_now, 0)

    def test_free_left_queue_span_is_measured_separately(self) -> None:
        main_lane = Zone("main_approach", ((0, 0), (100, 0), (100, 100), (0, 100)), "approach")
        free_left = Zone("free_left_lane", ((50, 0), (100, 0), (100, 100), (50, 100)), "free_left")
        analyzer = TrafficAnalyzer("approach", [main_lane, free_left])
        for timestamp in (0.0, 0.1):
            analyzer.update([
                Detection(1, "car", 0.9, (55, 10, 75, 30)),
                Detection(2, "car", 0.9, (80, 70, 100, 90)),
            ], timestamp)
        self.assertGreater(analyzer.metrics.current_queue_extent_pixels, 0)
        self.assertGreater(analyzer.metrics.queue_length_ratio_samples[-1], 0)


if __name__ == "__main__":
    unittest.main()
