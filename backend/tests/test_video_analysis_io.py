"""Video decoder smoke checks for clips with missing FPS metadata."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vision_engine.camera.video_analysis import analyze_video
from vision_engine.detection.yolo import Detection
from vision_engine.roi.zones import Zone


class CarDetector:
    def detect_and_track(self, frame: object) -> list[Detection]:
        return [Detection(7, "car", 0.95, (10, 10, 30, 30))]


class VideoInputRobustnessTests(unittest.TestCase):
    def test_missing_fps_metadata_uses_fallback_and_decodes_frames(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are unavailable")

        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "test.avi"
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
            if not writer.isOpened():
                self.skipTest("The local OpenCV build has no MJPG encoder")
            for frame_number in range(8):
                frame = np.full((48, 64, 3), frame_number * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            native_capture = cv2.VideoCapture
            previews: list[dict[str, object]] = []

            class MissingFpsCapture:
                def __init__(self, source: str) -> None:
                    self.inner = native_capture(source)

                def isOpened(self) -> bool:
                    return self.inner.isOpened()

                def get(self, prop: int) -> float:
                    if prop == cv2.CAP_PROP_FPS:
                        return 0.0
                    return self.inner.get(prop)

                def read(self):
                    return self.inner.read()

                def release(self) -> None:
                    self.inner.release()

            with patch.object(cv2, "VideoCapture", side_effect=MissingFpsCapture):
                result = analyze_video(
                    camera_id="approach",
                    source_path=clip,
                    output_path=Path(folder) / "unused.mp4",
                    zones=[
                        Zone("main_approach", ((0, 0), (63, 0), (63, 47), (0, 47)), "approach"),
                        Zone("free_left_lane", ((0, 0), (50, 0), (50, 47), (0, 47)), "free_left"),
                    ],
                    detector=CarDetector(),
                    save_annotated_video=False,
                    preview_callback=lambda _frame, counts: previews.append(counts),
                )

            self.assertEqual(result["frame_count_processed"], 8)
            self.assertEqual(result["inference_frame_count"], 8)
            self.assertTrue(result["input_warnings"])
            self.assertGreater(result["clip_duration_seconds"], 0)
            self.assertTrue(previews)
            self.assertGreater(max(int(item["free_left_vehicles_in_frame"]) for item in previews), 0)


if __name__ == "__main__":
    unittest.main()
