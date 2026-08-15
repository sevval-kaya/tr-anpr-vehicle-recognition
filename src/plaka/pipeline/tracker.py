"""Cross-frame vehicle tracking + plate-reading and vehicle-type consensus.

InferencePipeline is deliberately single-frame/stateless (see its own
docstring) — this is the "tracker-aware aggregator" that was always the
planned next layer once real video data existed to test it against
(roadmap stage 3+). It solves a problem single-frame reading can't: the
same real plate, read across several consecutive frames, often comes back
as several *different* wrong strings (e.g. "23 AI 638", "23 MC 633", "23
ACM 638" — all from the same physical plate a few frames apart, see
docs/decisions.md #33/#35) because OCR is right at the edge of what a
small/angled crop supports. No single frame's reading is trustworthy on
its own, but the same vehicle is now visible across enough frames (after
the #35 detector fine-tune widened the detection window) that voting
across them is worth doing.

Tracking is deliberately simple — greedy IoU matching against the last
few frames' boxes, not a real tracker library (Kalman filters, appearance
embeddings, etc.). vehicle_type is NEVER used as a matching criterion —
matching is IoU/position only. It still needs to be voted on, though: the
COCO vehicle detector applies NMS per-class, so it doesn't suppress an
overlapping "car" box against a "bus" box the way it would two overlapping
"car" boxes — on some real frames it emits *both* class labels for the
exact same physical object in the same frame (confirmed empirically on
arac3.mp4, see docs/decisions.md #37). Left alone, those duplicates
compete for the same track and can split one physical vehicle into two
parallel tracks purely because of a class-label flicker, which also
halves each track's plate-reading vote count. `_merge_same_frame_duplicates`
collapses those before matching; `VehicleTrack.consensus_vehicle_type`
(majority vote, tie-broken by confidence — same pattern as the plate-text
vote) is what actually gets displayed, so a flickering label never reaches
the user either.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from plaka.pipeline.schemas import BoundingBox, FrameResult, VehicleDetection

# Below this IoU, a detection in the new frame is treated as a different
# vehicle rather than a continuation of an existing track.
DEFAULT_MIN_IOU_FOR_MATCH = 0.3

# At or above this IoU, two detections *within the same frame* are treated
# as duplicate detections of one physical object (see module docstring) —
# much stricter than the cross-frame match threshold above, since there's
# no motion between them to account for.
DEFAULT_INTRA_FRAME_DUPLICATE_IOU = 0.7

# A track survives this many frame-indices without a match before being
# retired (not "1 frame" — --frame-stride/--sample-interval-seconds mean
# "the next processed frame" can be many raw frames later, and a vehicle
# can be missed for a frame or two without actually being a different one).
DEFAULT_MAX_FRAMES_SINCE_SEEN = 15


@dataclass(frozen=True, slots=True)
class PlateObservation:
    frame_index: int
    normalized_text: str
    ocr_confidence: float


@dataclass(frozen=True, slots=True)
class TypeObservation:
    vehicle_type: str
    detection_confidence: float


def _majority_vote(
    candidates: list[str], confidences: list[float]
) -> str:
    """Shared voting rule for both plate text and vehicle_type: majority
    count wins, ties broken by whichever candidate had the single highest
    confidence value.
    """
    counts = Counter(candidates)
    top_count = max(counts.values())
    tied = {value for value, count in counts.items() if count == top_count}
    if len(tied) == 1:
        return next(iter(tied))
    best_index = max(
        (i for i, value in enumerate(candidates) if value in tied),
        key=lambda i: confidences[i],
    )
    return candidates[best_index]


@dataclass
class VehicleTrack:
    track_id: int
    last_box: BoundingBox
    last_frame_index: int
    plate_observations: list[PlateObservation] = field(default_factory=list)
    type_observations: list[TypeObservation] = field(default_factory=list)

    def add_plate_observation(self, frame_index: int, normalized_text: str, ocr_confidence: float) -> None:
        self.plate_observations.append(PlateObservation(frame_index, normalized_text, ocr_confidence))

    def add_type_observation(self, vehicle_type: str, detection_confidence: float) -> None:
        self.type_observations.append(TypeObservation(vehicle_type, detection_confidence))

    @property
    def consensus_text(self) -> str | None:
        """Majority vote over every valid-format plate reading collected
        for this track, tie-broken by the single highest OCR confidence.
        Returns None until at least one valid reading has been seen.
        """
        if not self.plate_observations:
            return None
        return _majority_vote(
            [o.normalized_text for o in self.plate_observations],
            [o.ocr_confidence for o in self.plate_observations],
        )

    @property
    def consensus_vehicle_type(self) -> str:
        """Majority vote over every vehicle_type label this track has been
        seen with. Always has at least one observation (recorded when the
        track is created), so this is never empty in practice.
        """
        return _majority_vote(
            [o.vehicle_type for o in self.type_observations],
            [o.detection_confidence for o in self.type_observations],
        )

    @property
    def observation_count(self) -> int:
        return len(self.plate_observations)


def _merge_same_frame_duplicates(
    vehicles: list[VehicleDetection], duplicate_iou: float
) -> list[list[int]]:
    """Groups `vehicles` indices into clusters of near-certain duplicate
    detections of the same physical object within one frame (see module
    docstring) via union-find on pairwise IoU >= `duplicate_iou`. A vehicle
    with no duplicates is its own single-element cluster.
    """
    n = len(vehicles)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            if vehicles[i].box.iou(vehicles[j].box) >= duplicate_iou:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


class VehicleTracker:
    """Owns the set of active tracks across a video's frames. Call
    `update()` once per processed frame, in frame_index order.
    """

    def __init__(
        self,
        min_iou_for_match: float = DEFAULT_MIN_IOU_FOR_MATCH,
        intra_frame_duplicate_iou: float = DEFAULT_INTRA_FRAME_DUPLICATE_IOU,
        max_frames_since_seen: int = DEFAULT_MAX_FRAMES_SINCE_SEEN,
    ) -> None:
        self._min_iou_for_match = min_iou_for_match
        self._intra_frame_duplicate_iou = intra_frame_duplicate_iou
        self._max_frames_since_seen = max_frames_since_seen
        self._tracks: dict[int, VehicleTrack] = {}
        self._next_track_id = 1

    def update(self, frame_index: int, vehicles: list[VehicleDetection]) -> list[int]:
        """Match this frame's `vehicles` (in order) against active tracks
        by IoU alone (never by vehicle_type — see module docstring) —
        each *cluster* of mutual-duplicate detections matches at most the
        single most-overlapping still-unclaimed track, greedily. Unmatched
        clusters start new tracks. Records a plate observation (when the
        vehicle has a valid-format reading) and a type observation (always)
        on the matched/new track for every vehicle in the cluster, not
        just its representative.

        Returns one track_id per vehicle, same order as `vehicles` — every
        vehicle in a duplicate cluster gets the same track_id.
        """
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame_index > self._max_frames_since_seen
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

        clusters = _merge_same_frame_duplicates(vehicles, self._intra_frame_duplicate_iou)
        # One representative per cluster (highest detection_confidence)
        # stands in for the whole cluster during track matching.
        representative_index = {
            id(cluster): max(cluster, key=lambda i: vehicles[i].detection_confidence)
            for cluster in clusters
        }

        candidate_pairs: list[tuple[float, int, int]] = []  # (iou, cluster_index, track_id)
        for cluster_index, cluster in enumerate(clusters):
            rep = vehicles[representative_index[id(cluster)]]
            for track_id, track in self._tracks.items():
                iou = rep.box.iou(track.last_box)
                if iou >= self._min_iou_for_match:
                    candidate_pairs.append((iou, cluster_index, track_id))
        candidate_pairs.sort(key=lambda pair: pair[0], reverse=True)

        assigned_track_id: dict[int, int] = {}
        claimed_tracks: set[int] = set()
        for _iou, cluster_index, track_id in candidate_pairs:
            if cluster_index in assigned_track_id or track_id in claimed_tracks:
                continue
            assigned_track_id[cluster_index] = track_id
            claimed_tracks.add(track_id)

        track_ids = [0] * len(vehicles)
        for cluster_index, cluster in enumerate(clusters):
            rep = vehicles[representative_index[id(cluster)]]
            track_id = assigned_track_id.get(cluster_index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = VehicleTrack(
                    track_id=track_id, last_box=rep.box, last_frame_index=frame_index
                )
            else:
                track = self._tracks[track_id]
                track.last_box = rep.box
                track.last_frame_index = frame_index

            track = self._tracks[track_id]
            for vehicle_index in cluster:
                vehicle = vehicles[vehicle_index]
                track.add_type_observation(vehicle.vehicle_type, vehicle.detection_confidence)
                plate = vehicle.plate
                if plate is not None and plate.is_format_valid and plate.normalized_text:
                    track.add_plate_observation(frame_index, plate.normalized_text, plate.ocr_confidence)
                track_ids[vehicle_index] = track_id

        return track_ids

    def get_track(self, track_id: int) -> VehicleTrack | None:
        return self._tracks.get(track_id)


def apply_consensus(result: FrameResult, tracker: VehicleTracker, frame_index: int) -> FrameResult:
    """Runs `tracker.update()` for this frame and returns a new FrameResult
    where each vehicle's `track_id` is set, its `vehicle_type` is replaced
    by its track's cross-frame consensus type, and — once its track has
    seen at least one valid-format reading — its plate's `normalized_text`
    is replaced by the track's consensus text. Box/confidence values are
    left untouched.
    """
    track_ids = tracker.update(frame_index, result.vehicles)
    updated_vehicles = []
    for vehicle, track_id in zip(result.vehicles, track_ids, strict=True):
        track = tracker.get_track(track_id)
        consensus_text = track.consensus_text if track is not None else None
        consensus_type = track.consensus_vehicle_type if track is not None else vehicle.vehicle_type
        if vehicle.plate is not None and consensus_text is not None:
            updated_plate = vehicle.plate.model_copy(update={"normalized_text": consensus_text})
        else:
            updated_plate = vehicle.plate
        updated_vehicles.append(
            vehicle.model_copy(
                update={"plate": updated_plate, "track_id": track_id, "vehicle_type": consensus_type}
            )
        )
    return result.model_copy(update={"vehicles": updated_vehicles})
