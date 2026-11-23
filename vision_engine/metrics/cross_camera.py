"""Conservative, privacy-preserving association of camera-local vehicle tracks.

This is a classical appearance-and-time matcher, not a guaranteed vehicle identity
system. Uncertain pairs are deliberately left unmatched and reported as a range.
Only short-lived color histograms and keypoint descriptors are used; no crops or
plate text are retained or written to disk.
"""

from __future__ import annotations

from typing import Any
import math


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / denom if denom else 0.0


def associate_camera_tracks(camera_results: list[dict[str, Any]], threshold: float = 0.90) -> dict[str, Any]:
    """Greedily match high-confidence cross-camera tracks with overlapping time.

    A one-to-one constraint within each camera pair prevents a single detection
    from collapsing several vehicles into one identity. `threshold` is purposely
    high because color appearance alone cannot safely resolve ambiguous vehicles.
    """
    tracks: list[dict[str, Any]] = []
    for result in camera_results:
        for track in result.pop("_vehicle_tracklets", []):
            if track.get("samples", 0) >= 2 and track.get("color_hist"):
                tracks.append({**track, "camera_id": result.get("camera_id")})

    parents = list(range(len(tracks)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    pair_scores: list[tuple[float, int, int]] = []
    for i, a in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            b = tracks[j]
            if a["camera_id"] == b["camera_id"] or a.get("category") != b.get("category"):
                continue
            # This prototype assumes clips share a clock. Require overlapping
            # track intervals; unsynchronized feeds need calibration first.
            if max(a["start_s"], b["start_s"]) > min(a["end_s"], b["end_s"]) + 0.35:
                continue
            color = _cosine(a["color_hist"], b["color_hist"])
            # Normalized HSV histograms are view-sensitive. Strong color agreement
            # is evidence, never proof, so avoid forcing low-confidence matches.
            score = color
            if score >= threshold:
                pair_scores.append((score, i, j))
    # Abstain if a track has a similarly colored competing candidate in the
    # same other camera. Color-only similarity cannot resolve two lookalikes.
    candidates: list[tuple[float, int, int]] = []
    ambiguous_candidate_pairs: set[tuple[int, int]] = set()
    for score, i, j in pair_scores:
        camera_i, camera_j = tracks[i]["camera_id"], tracks[j]["camera_id"]
        second_i = max((s for s, x, y in pair_scores if x == i and y != j and tracks[y]["camera_id"] == camera_j), default=0.0)
        second_j = max((s for s, x, y in pair_scores if y == j and x != i and tracks[x]["camera_id"] == camera_i), default=0.0)
        if score - second_i >= 0.05 and score - second_j >= 0.05:
            candidates.append((score, i, j))
        else:
            ambiguous_candidate_pairs.add((i, j))
    candidates.sort(reverse=True)
    linked_pairs: list[dict[str, Any]] = []
    cluster_cameras = [{t["camera_id"]} for t in tracks]
    for score, i, j in candidates:
        a, b = tracks[i], tracks[j]
        ra, rb = find(i), find(j)
        if ra == rb or cluster_cameras[ra].intersection(cluster_cameras[rb]):
            continue
        cluster_cameras[ra] |= cluster_cameras[rb]
        union(i, j)
        linked_pairs.append({"camera_a": a["camera_id"], "track_a": a["track_id"], "camera_b": b["camera_id"], "track_b": b["track_id"], "appearance_similarity": round(score, 3)})

    unique_ids = {find(i) for i in range(len(tracks))}
    observed_tracks = sum(int(r.get("vehicle_track_count", 0)) for r in camera_results)
    confirmed_duplicates = len(tracks) - len(unique_ids)
    # Tracks without sufficient image evidence and all ambiguous candidates remain
    # separate. The upper bound is the per-camera total, not a fabricated exact ID.
    return {
        "method": "synchronized_track_overlap_and_vehicle_color_histogram",
        "confirmed_cross_camera_matches": linked_pairs,
        "matched_tracklets": len(tracks),
        "unmatched_or_unusable_tracklets": max(0, observed_tracks - len(tracks)),
        "ambiguous_candidate_pairs": len(ambiguous_candidate_pairs),
        "unique_vehicle_count_estimate_after_confirmed_merges": max(0, observed_tracks - confirmed_duplicates),
        "unique_vehicle_count_upper_bound": observed_tracks,
        "identity_confidence": "low_to_moderate; ambiguous matches are not merged",
        "assumptions": ["Camera clips are synchronized to a common clock.", "Similar vehicle color is weak evidence and can produce false matches; review matches before consequential decisions."],
    }


class LiveCrossCameraAssociator:
    """Maintain session-local identities for simultaneously visible USB views."""

    def __init__(self, threshold: float = 0.94, active_window_s: float = 2.0) -> None:
        self.threshold = threshold
        self.active_window_s = active_window_s
        self.tracks: dict[tuple[str, int], dict[str, Any]] = {}
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}
        self.cluster_cameras: dict[tuple[str, int], set[str]] = {}
        self.matches: list[dict[str, Any]] = []
        self.pending: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}
        self.ambiguous_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()

    def _find(self, key: tuple[str, int]) -> tuple[str, int]:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def observe(self, camera_id: str, track: dict[str, Any]) -> dict[str, Any]:
        track_id = int(track["track_id"])
        key = (camera_id, track_id)
        previous = self.tracks.get(key)
        if previous is None:
            self.parent[key] = key
            self.cluster_cameras[key] = {camera_id}
            self.tracks[key] = dict(track)
        else:
            old_samples = int(previous.get("samples", 1))
            new_samples = int(track.get("samples", old_samples))
            if new_samples > old_samples:
                old_hist, new_hist = previous.get("color_hist", []), track.get("color_hist", [])
                if len(old_hist) == len(new_hist) and new_hist:
                    self.tracks[key]["color_hist"] = [
                        (a * old_samples + b * (new_samples - old_samples)) / new_samples
                        for a, b in zip(old_hist, new_hist)
                    ]
            self.tracks[key]["samples"] = max(old_samples, new_samples)
            self.tracks[key]["end_s"] = max(float(previous.get("end_s", 0)), float(track.get("end_s", 0)))

        active = [
            (other_key, other)
            for other_key, other in self.tracks.items()
            if other_key != key
            and other_key[0] != camera_id
            and other.get("category") == track.get("category")
            and int(other.get("samples", 0)) >= 2
            and float(track.get("end_s", 0)) - float(other.get("end_s", 0)) <= self.active_window_s
            and float(other.get("end_s", 0)) - float(track.get("end_s", 0)) <= self.active_window_s
        ]
        current = self.tracks[key]
        current_time = float(current.get("end_s", 0))
        self.pending = {pair: started for pair, started in self.pending.items() if current_time - started <= self.active_window_s * 2}
        if int(current.get("samples", 0)) >= 2 and current.get("color_hist"):
            scored = [(other_key, _cosine(current["color_hist"], other["color_hist"])) for other_key, other in active if other.get("color_hist")]
            scored = [(other_key, score) for other_key, score in scored if score >= self.threshold]
            best_by_camera: list[tuple[tuple[str, int], float]] = []
            for other_camera in {other_key[0] for other_key, _ in scored}:
                choices = sorted((pair for pair in scored if pair[0][0] == other_camera), key=lambda pair: pair[1], reverse=True)
                if len(choices) == 1 or choices[0][1] - choices[1][1] >= 0.05:
                    best_by_camera.append(choices[0])
                else:
                    tied = choices[:2]
                    self.ambiguous_pairs.add(tuple(sorted((tied[0][0], tied[1][0]))))
                    self.pending.clear()
            best_by_camera.sort(key=lambda pair: pair[1], reverse=True)
            if best_by_camera:
                other_key, score = best_by_camera[0]
                candidate_track = self.tracks[other_key]
                same_camera_lookalike = any(
                    local_key != key
                    and local_key[0] == camera_id
                    and abs(current_time - float(local_track.get("end_s", 0))) <= self.active_window_s
                    and int(local_track.get("samples", 0)) >= 2
                    and _cosine(candidate_track.get("color_hist", []), local_track.get("color_hist", [])) >= self.threshold
                    for local_key, local_track in self.tracks.items()
                )
                if same_camera_lookalike:
                    self.ambiguous_pairs.add(tuple(sorted((key, other_key))))
                    self.pending.clear()
                    return self.summary()
                # Let a short ambiguity window pass before merging. If a second
                # same-looking local track appears in that interval, abstain.
                pair = tuple(sorted((key, other_key)))
                competing_pending = any(
                    other_key in pending_pair and pending_pair != pair
                    and any(member[0] == key[0] for member in pending_pair if member != other_key)
                    for pending_pair in self.pending
                )
                if competing_pending:
                    self.pending.clear()
                    return self.summary()
                proposed_at = self.pending.setdefault(pair, float(current.get("end_s", 0)))
                if float(current.get("end_s", 0)) - proposed_at < 1.0:
                    return self.summary()
                root_a, root_b = self._find(key), self._find(other_key)
                if root_a != root_b and not self.cluster_cameras[root_a].intersection(self.cluster_cameras[root_b]):
                    self.parent[root_b] = root_a
                    self.cluster_cameras[root_a] |= self.cluster_cameras.pop(root_b)
                    self.matches.append({"camera_a": key[0], "track_a": key[1], "camera_b": other_key[0], "track_b": other_key[1], "appearance_similarity": round(score, 3)})
                    self.pending.pop(pair, None)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        roots = {self._find(item) for item in self.parent}
        return {
            "estimated_unique_vehicles": len(roots),
            "local_vehicle_tracklets": len(self.tracks),
            "confirmed_cross_camera_matches": list(self.matches[-50:]),
            "ambiguous_candidate_pairs": len(self.ambiguous_pairs),
            "identity_confidence": "experimental appearance matches; uncertain pairs remain separate",
        }
