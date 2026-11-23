"""Validated zone configuration and point-in-polygon helpers."""

from dataclasses import dataclass
from typing import Sequence

Point = tuple[int, int]


@dataclass(frozen=True)
class Zone:
    """A named polygon configured in source-frame pixel coordinates."""

    name: str
    polygon: tuple[Point, ...]
    kind: str

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"Zone '{self.name}' needs at least three polygon points")
        if not self.name.strip():
            raise ValueError("Zone name cannot be empty")

    def contains(self, point: Point) -> bool:
        """Return true for points inside or directly on the polygon boundary."""

        inside = False
        x, y = point
        points = self.polygon
        previous_x, previous_y = points[-1]
        for current_x, current_y in points:
            cross = (x - previous_x) * (current_y - previous_y) - (y - previous_y) * (current_x - previous_x)
            if cross == 0 and min(previous_x, current_x) <= x <= max(previous_x, current_x) and min(previous_y, current_y) <= y <= max(previous_y, current_y):
                return True
            if (current_y > y) != (previous_y > y):
                intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
                if x < intersection_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside


def zones_from_mapping(raw: Sequence[dict[str, object]]) -> list[Zone]:
    """Parse the simple JSON zone format used by the command-line runner."""

    zones: list[Zone] = []
    for item in raw:
        points = tuple((int(point[0]), int(point[1])) for point in item["polygon"])  # type: ignore[index]
        zones.append(Zone(name=str(item["name"]), polygon=points, kind=str(item["kind"])))
    return zones
