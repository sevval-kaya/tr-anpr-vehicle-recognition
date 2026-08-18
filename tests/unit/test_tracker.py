from plaka.pipeline.schemas import BoundingBox, FrameResult, PlateReading, VehicleDetection
from plaka.pipeline.tracker import VehicleTracker, apply_consensus


def _vehicle(
    box: BoundingBox,
    plate_text: str | None = None,
    ocr_confidence: float = 0.8,
    is_valid: bool = True,
    vehicle_type: str = "car",
) -> VehicleDetection:
    plate = None
    if plate_text is not None:
        plate = PlateReading(
            box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
            raw_text=plate_text.replace(" ", ""),
            normalized_text=plate_text if is_valid else None,
            is_format_valid=is_valid,
            detection_confidence=0.9,
            ocr_confidence=ocr_confidence,
        )
    return VehicleDetection(
        box=box, vehicle_type=vehicle_type, detection_confidence=0.9, plate=plate
    )


class TestTrackMatching:
    def test_overlapping_box_in_next_frame_gets_same_track_id(self) -> None:
        tracker = VehicleTracker()
        box1 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        box2 = BoundingBox(x_min=5, y_min=5, x_max=105, y_max=105)  # heavily overlapping

        ids_frame0 = tracker.update(0, [_vehicle(box1)])
        ids_frame1 = tracker.update(1, [_vehicle(box2)])

        assert ids_frame0 == ids_frame1

    def test_non_overlapping_boxes_get_different_track_ids(self) -> None:
        tracker = VehicleTracker()
        box1 = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        box2 = BoundingBox(x_min=500, y_min=500, x_max=550, y_max=550)

        ids_frame0 = tracker.update(0, [_vehicle(box1)])
        ids_frame1 = tracker.update(1, [_vehicle(box2)])

        assert ids_frame0 != ids_frame1

    def test_two_vehicles_matched_by_best_iou_not_array_order(self) -> None:
        tracker = VehicleTracker()
        left = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        right = BoundingBox(x_min=200, y_min=200, x_max=250, y_max=250)
        left_id, right_id = tracker.update(0, [_vehicle(left), _vehicle(right)])

        # Frame 1: same two vehicles, but listed in reversed order — each
        # should still match its own previous box via IoU, not by index.
        left_moved = BoundingBox(x_min=5, y_min=5, x_max=55, y_max=55)
        right_moved = BoundingBox(x_min=205, y_min=205, x_max=255, y_max=255)
        ids_frame1 = tracker.update(1, [_vehicle(right_moved), _vehicle(left_moved)])

        assert ids_frame1[0] == right_id
        assert ids_frame1[1] == left_id

    def test_track_retires_after_max_frames_since_seen(self) -> None:
        tracker = VehicleTracker(max_frames_since_seen=2)
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        ids_frame0 = tracker.update(0, [_vehicle(box)])
        # Gap of more than 2 frames with no detections in between.
        ids_frame_later = tracker.update(10, [_vehicle(box)])
        assert ids_frame0 != ids_frame_later

    def test_track_survives_a_short_gap(self) -> None:
        tracker = VehicleTracker(max_frames_since_seen=5)
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        ids_frame0 = tracker.update(0, [_vehicle(box)])
        ids_frame3 = tracker.update(3, [_vehicle(box)])
        assert ids_frame0 == ids_frame3

    def test_vehicle_with_no_plate_still_gets_tracked(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        ids = tracker.update(0, [_vehicle(box, plate_text=None)])
        track = tracker.get_track(ids[0])
        assert track is not None
        assert track.consensus_text is None


class TestConsensus:
    def test_majority_vote_wins(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        tracker.update(0, [_vehicle(box, "23 ACM 638", ocr_confidence=0.5)])
        tracker.update(1, [_vehicle(box, "23 AI 638", ocr_confidence=0.9)])
        tracker.update(2, [_vehicle(box, "23 ACM 638", ocr_confidence=0.6)])

        track_id = tracker.update(3, [_vehicle(box, "23 ACM 638", ocr_confidence=0.4)])[0]
        assert tracker.get_track(track_id).consensus_text == "23 ACM 638"

    def test_tie_broken_by_highest_confidence(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        tracker.update(0, [_vehicle(box, "23 AAA 111", ocr_confidence=0.3)])
        track_id = tracker.update(1, [_vehicle(box, "23 BBB 222", ocr_confidence=0.9)])[0]
        assert tracker.get_track(track_id).consensus_text == "23 BBB 222"

    def test_invalid_format_readings_are_excluded_from_the_vote(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        tracker.update(0, [_vehicle(box, "GARBLED", is_valid=False)])
        track_id = tracker.update(1, [_vehicle(box, "23 ACM 638", is_valid=True)])[0]
        track = tracker.get_track(track_id)
        assert track.observation_count == 1
        assert track.consensus_text == "23 ACM 638"

    def test_no_observations_yields_none(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        track_id = tracker.update(0, [_vehicle(box, plate_text=None)])[0]
        assert tracker.get_track(track_id).consensus_text is None


class TestIntraFrameDuplicates:
    """Reproduces the real bug found on arac3.mp4 (docs/decisions.md #37):
    the COCO vehicle detector applies NMS per-class, so on some frames it
    emits *both* a "car" and a "bus" box for the same physical vehicle —
    two entries in the same frame's vehicle list with near-identical
    boxes. Left unmerged, these compete for the same track and can split
    one vehicle into two, each getting only half the plate-reading votes.
    """

    def test_duplicate_car_and_bus_boxes_in_one_frame_become_one_track(self) -> None:
        tracker = VehicleTracker()
        car_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
        bus_box = BoundingBox(x_min=102, y_min=101, x_max=201, y_max=199)  # near-identical
        ids = tracker.update(0, [_vehicle(car_box, vehicle_type="car"), _vehicle(bus_box, vehicle_type="bus")])
        assert ids[0] == ids[1]
        # A second, unrelated vehicle in the very next frame must get a
        # genuinely new id (2) rather than 3 — proving only one track (not
        # a phantom second one from the duplicate) was created in frame 0.
        other_box = BoundingBox(x_min=900, y_min=900, x_max=950, y_max=950)
        next_id = tracker.update(1, [_vehicle(other_box)])[0]
        assert next_id != ids[0]
        assert next_id == 2

    def test_type_flicker_across_frames_does_not_split_the_track(self) -> None:
        # frame0: car only. frame1: car+bus duplicate (the real failure
        # mode). frame2: bus only. All three must stay one track.
        tracker = VehicleTracker()
        box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
        box_shifted = BoundingBox(x_min=103, y_min=102, x_max=203, y_max=202)

        id0 = tracker.update(0, [_vehicle(box, vehicle_type="car")])[0]
        ids1 = tracker.update(
            1, [_vehicle(box_shifted, vehicle_type="car"), _vehicle(box_shifted, vehicle_type="bus")]
        )
        id2 = tracker.update(2, [_vehicle(box_shifted, vehicle_type="bus")])[0]

        assert ids1[0] == id0
        assert ids1[1] == id0
        assert id2 == id0

    def test_genuinely_distinct_nearby_vehicles_are_not_merged(self) -> None:
        # Two real, separate vehicles that happen to be close but don't
        # overlap heavily — must NOT be merged into one track.
        tracker = VehicleTracker()
        left = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        right = BoundingBox(x_min=90, y_min=0, x_max=190, y_max=100)  # ~10% overlap only
        ids = tracker.update(0, [_vehicle(left), _vehicle(right)])
        assert ids[0] != ids[1]

    def test_plate_reading_from_either_duplicate_counts_toward_the_shared_track(self) -> None:
        # Only the "bus"-labeled duplicate has a plate reading this frame
        # (e.g. only one of the two boxes' crops happened to OCR
        # successfully) — it must not be lost just because the "car"
        # duplicate (with no reading) was picked as the match representative.
        tracker = VehicleTracker()
        car_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
        bus_box = BoundingBox(x_min=101, y_min=100, x_max=201, y_max=200)
        ids = tracker.update(
            0,
            [
                _vehicle(car_box, plate_text=None, vehicle_type="car"),
                _vehicle(bus_box, "23 ACM 638", vehicle_type="bus"),
            ],
        )
        track = tracker.get_track(ids[0])
        assert track.consensus_text == "23 ACM 638"


class TestVehicleTypeConsensus:
    def test_majority_type_wins(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        tracker.update(0, [_vehicle(box, vehicle_type="car")])
        tracker.update(1, [_vehicle(box, vehicle_type="bus")])
        track_id = tracker.update(2, [_vehicle(box, vehicle_type="car")])[0]
        assert tracker.get_track(track_id).consensus_vehicle_type == "car"

    def test_apply_consensus_overwrites_the_displayed_type(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        apply_consensus(FrameResult(frame_index=0, vehicles=[_vehicle(box, vehicle_type="car")]), tracker, 0)
        apply_consensus(FrameResult(frame_index=1, vehicles=[_vehicle(box, vehicle_type="car")]), tracker, 1)
        result2 = FrameResult(frame_index=2, vehicles=[_vehicle(box, vehicle_type="bus")])
        updated2 = apply_consensus(result2, tracker, 2)
        # Single-frame label at frame 2 is "bus", but the track has seen
        # "car" twice and "bus" once — displayed type should be the majority.
        assert updated2.vehicles[0].vehicle_type == "car"


class TestSpeedEstimation:
    def test_position_history_accumulates_across_updates(self) -> None:
        tracker = VehicleTracker()
        box0 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        box1 = BoundingBox(x_min=10, y_min=0, x_max=110, y_max=50)
        track_id = tracker.update(0, [_vehicle(box0)], timestamp_seconds=0.0)[0]
        tracker.update(1, [_vehicle(box1)], timestamp_seconds=1.0)
        track = tracker.get_track(track_id)
        assert track is not None
        assert len(track.position_history) == 2

    def test_estimated_speed_kmh_is_none_with_a_single_observation(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        track_id = tracker.update(0, [_vehicle(box)], timestamp_seconds=0.0)[0]
        assert tracker.get_track(track_id).estimated_speed_kmh is None

    def test_estimated_speed_kmh_is_a_plausible_positive_number_for_a_moving_vehicle(self) -> None:
        tracker = VehicleTracker()
        # Box shifts 10px right per second, constant 100px width -> a real,
        # small but nonzero speed, never 0-200 km/h-implausible.
        track_id = None
        for i in range(5):
            box = BoundingBox(x_min=i * 10, y_min=0, x_max=i * 10 + 100, y_max=50)
            ids = tracker.update(i, [_vehicle(box)], timestamp_seconds=float(i))
            track_id = ids[0]
        speed = tracker.get_track(track_id).estimated_speed_kmh
        assert speed is not None
        assert 0 < speed < 200

    def test_stationary_vehicle_estimates_near_zero_speed(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        track_id = None
        for i in range(4):
            ids = tracker.update(i, [_vehicle(box)], timestamp_seconds=float(i))
            track_id = ids[0]
        assert tracker.get_track(track_id).estimated_speed_kmh == 0.0

    def test_apply_consensus_sets_estimated_speed_kmh(self) -> None:
        tracker = VehicleTracker()
        box0 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        box1 = BoundingBox(x_min=50, y_min=0, x_max=150, y_max=50)
        result0 = FrameResult(frame_index=0, vehicles=[_vehicle(box0)])
        apply_consensus(result0, tracker, 0, timestamp_seconds=0.0)
        result1 = FrameResult(frame_index=1, vehicles=[_vehicle(box1)])
        updated1 = apply_consensus(result1, tracker, 1, timestamp_seconds=1.0)
        assert updated1.vehicles[0].estimated_speed_kmh is not None
        assert updated1.vehicles[0].estimated_speed_kmh > 0

    def test_omitting_timestamp_seconds_does_not_crash(self) -> None:
        # Callers that don't pass timestamp_seconds (most existing tests)
        # must keep working — frame_index is used as a time-base fallback.
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        tracker.update(0, [_vehicle(box)])
        track_id = tracker.update(1, [_vehicle(box)])[0]
        # No assertion on the numeric value (the fallback time base is
        # meaningless) — just that it doesn't raise.
        _ = tracker.get_track(track_id).estimated_speed_kmh


class TestApplyConsensus:
    def test_replaces_plate_text_once_a_track_has_a_reading(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)

        result0 = FrameResult(frame_index=0, vehicles=[_vehicle(box, "23 ACM 638")])
        updated0 = apply_consensus(result0, tracker, 0)
        assert updated0.vehicles[0].plate.normalized_text == "23 ACM 638"
        assert updated0.vehicles[0].track_id is not None

        result1 = FrameResult(frame_index=1, vehicles=[_vehicle(box, "23 AI 638")])
        updated1 = apply_consensus(result1, tracker, 1)
        result2 = FrameResult(frame_index=2, vehicles=[_vehicle(box, "23 ACM 638")])
        updated2 = apply_consensus(result2, tracker, 2)

        # Single-frame reading in frame 1 was the noisy "23 AI 638", but by
        # frame 2 the track has seen "23 ACM 638" twice vs "23 AI 638" once
        # — the displayed text should already reflect that majority, not
        # frame 2's own (matching) single-frame reading coincidentally.
        assert updated2.vehicles[0].plate.normalized_text == "23 ACM 638"
        assert updated1.vehicles[0].track_id == updated0.vehicles[0].track_id
        assert updated2.vehicles[0].track_id == updated0.vehicles[0].track_id

    def test_leaves_box_and_confidence_untouched(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=1, y_min=2, x_max=51, y_max=52)
        result = FrameResult(frame_index=0, vehicles=[_vehicle(box, "23 ACM 638")])
        updated = apply_consensus(result, tracker, 0)
        assert updated.vehicles[0].box == box
        assert updated.vehicles[0].detection_confidence == 0.9
        assert updated.vehicles[0].plate.box == result.vehicles[0].plate.box

    def test_vehicle_with_no_plate_is_untouched_besides_track_id(self) -> None:
        tracker = VehicleTracker()
        box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        result = FrameResult(frame_index=0, vehicles=[_vehicle(box, plate_text=None)])
        updated = apply_consensus(result, tracker, 0)
        assert updated.vehicles[0].plate is None
        assert updated.vehicles[0].track_id is not None
