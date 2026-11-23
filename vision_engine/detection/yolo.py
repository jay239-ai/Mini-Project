"""YOLO object detection, tracking, and the shared detection data type."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Detection:
    """One detected object and its camera-local tracking ID."""

    track_id: int | None
    category: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def footpoint(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (round((x1 + x2) / 2), round(y2))


class Detector(Protocol):
    def detect_and_track(self, frame: object) -> list[Detection]: ...


class TrafficModel(Protocol):
    names: dict[int, str]

    def track(self, source: object, **kwargs: object) -> list[object]: ...

TRAFFIC_CLASS_IDS = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck (COCO)


class UltralyticsByteTrackDetector:
    """Load one model once and preserve tracking state for one video stream."""

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        tracker: str = "bytetrack.yaml",
        confidence: float = 0.25,
        image_size: int = 640,
        device: str | None = None,
        model: TrafficModel | None = None,
    ) -> None:
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("Install project dependencies before running video analysis") from error
            model = YOLO(weights)
        self.model = model
        self.tracker = tracker
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def detect_and_track(self, frame: object) -> list[Detection]:
        """Detect traffic objects and attach stream-local ByteTrack IDs."""

        options: dict[str, Any] = {
            "persist": True,
            "tracker": self.tracker,
            "classes": TRAFFIC_CLASS_IDS,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device:
            options["device"] = self.device
            options["half"] = self.device != "cpu"
        results = self.model.track(frame, **options)
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        ids_tensor = getattr(boxes, "id", None)
        track_ids = ids_tensor.int().cpu().tolist() if ids_tensor is not None else [None] * len(coordinates)
        names = self.model.names
        return [
            Detection(
                track_id=int(track_id) if track_id is not None else None,
                category=str(names[int(class_id)]),
                confidence=float(confidence),
                box=tuple(float(value) for value in box),  # type: ignore[arg-type]
            )
            for box, class_id, confidence, track_id in zip(coordinates, classes, confidences, track_ids)
        ]
