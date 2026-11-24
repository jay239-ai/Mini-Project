from __future__ import annotations

import unittest
import threading

import numpy as np

from vision_engine.camera.live_capture import BlackCameraFeedError, LatestFrameSource, open_usb_camera


class FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False
        self.settings: list[tuple[int, int]] = []

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: int) -> None:
        self.settings.append((prop, value))


class FakeCV2:
    CAP_DSHOW = 1
    CAP_MSMF = 2
    CAP_ANY = 0
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, frames_by_backend: dict[int, np.ndarray]) -> None:
        self.frames_by_backend = frames_by_backend
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, _index: int, backend: int) -> FakeCapture:
        capture = FakeCapture(self.frames_by_backend[backend])
        self.captures.append(capture)
        return capture


class UsbCameraBackendTests(unittest.TestCase):
    def test_tries_next_windows_backend_when_first_returns_black(self) -> None:
        cv2 = FakeCV2({
            FakeCV2.CAP_DSHOW: np.zeros((12, 16, 3), dtype=np.uint8),
            FakeCV2.CAP_MSMF: np.full((12, 16, 3), 120, dtype=np.uint8),
            FakeCV2.CAP_ANY: np.full((12, 16, 3), 120, dtype=np.uint8),
        })

        capture, backend = open_usb_camera(cv2, 1)

        self.assertEqual(backend, FakeCV2.CAP_MSMF)
        self.assertTrue(cv2.captures[0].released)
        self.assertFalse(capture.released)

    def test_reports_when_every_backend_only_returns_black_frames(self) -> None:
        cv2 = FakeCV2({
            backend: np.zeros((12, 16, 3), dtype=np.uint8)
            for backend in (FakeCV2.CAP_DSHOW, FakeCV2.CAP_MSMF, FakeCV2.CAP_ANY)
        })

        with self.assertRaisesRegex(BlackCameraFeedError, "only black frames"):
            open_usb_camera(cv2, 1)

    def test_closing_one_source_does_not_stop_shared_camera_session(self) -> None:
        source = LatestFrameSource.__new__(LatestFrameSource)
        source.stop_event = threading.Event()
        source._close_event = threading.Event()
        source.capture = FakeCapture(np.full((2, 2, 3), 80, dtype=np.uint8))

        class FinishedThread:
            def join(self, timeout: float) -> None:
                return None

        source.thread = FinishedThread()
        source.close()

        self.assertFalse(source.stop_event.is_set())
        self.assertTrue(source._close_event.is_set())
        self.assertTrue(source.capture.released)


if __name__ == "__main__":
    unittest.main()
