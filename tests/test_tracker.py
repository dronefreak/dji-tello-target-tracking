"""Unit tests for the tracker module.

Tests centroid tracking, object association, and trajectory management.
"""

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.detector import Detection
from src.tracker import TrackedObject, ObjectTracker, SingleObjectTracker


class TestTrackedObject:
    """Test TrackedObject class."""

    def test_creation(self):
        """Test creating a tracked object."""
        detection = Detection(
            bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"
        )

        obj = TrackedObject(1, detection, max_history=10)

        assert obj.id == 1
        assert obj.class_name == "person"
        assert obj.bbox == (100, 100, 200, 200)
        assert obj.center == (150, 150)
        assert obj.confidence == 0.9
        assert obj.age == 1
        assert obj.disappeared == 0
        assert len(obj.centers) == 1
        assert len(obj.bboxes) == 1

    def test_update(self):
        """Test updating tracked object with new detection."""
        detection1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        obj = TrackedObject(1, detection1)

        detection2 = Detection((110, 110, 210, 210), 0.85, 0, "person")
        obj.update(detection2)

        assert obj.bbox == (110, 110, 210, 210)
        assert obj.center == (160, 160)
        assert obj.confidence == 0.85
        assert obj.age == 2
        assert obj.disappeared == 0
        assert len(obj.centers) == 2

    def test_mark_disappeared(self):
        """Test marking object as disappeared."""
        detection = Detection((100, 100, 200, 200), 0.9, 0, "person")
        obj = TrackedObject(1, detection)

        obj.mark_disappeared()
        assert obj.disappeared == 1

        obj.mark_disappeared()
        assert obj.disappeared == 2

    def test_velocity_calculation(self):
        """Test velocity calculation from trajectory."""
        detection1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        obj = TrackedObject(1, detection1)

        # Single point - no velocity
        velocity = obj.get_velocity()
        assert velocity is None

        # Add more points
        detection2 = Detection((110, 110, 210, 210), 0.9, 0, "person")
        obj.update(detection2)

        detection3 = Detection((120, 120, 220, 220), 0.9, 0, "person")
        obj.update(detection3)

        velocity = obj.get_velocity()
        assert velocity is not None
        vx, vy = velocity
        assert vx > 0  # Moving right
        assert vy > 0  # Moving down

    def test_predict_next_position(self):
        """Test position prediction based on velocity."""
        detection1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        obj = TrackedObject(1, detection1)

        detection2 = Detection((110, 100, 210, 200), 0.9, 0, "person")
        obj.update(detection2)

        predicted = obj.predict_next_position()
        assert predicted is not None
        assert predicted[0] > 160  # Should predict movement to the right

    def test_history_limit(self):
        """Test that history is limited to max_history."""
        detection = Detection((100, 100, 200, 200), 0.9, 0, "person")
        obj = TrackedObject(1, detection, max_history=5)

        # Add 10 updates
        for i in range(10):
            det = Detection((100 + i * 10, 100, 200 + i * 10, 200), 0.9, 0, "person")
            obj.update(det)

        # Should only keep last 5
        assert len(obj.centers) == 5
        assert len(obj.bboxes) == 5


class TestObjectTracker:
    """Test ObjectTracker class."""

    def test_creation(self):
        """Test creating a tracker."""
        config = Config()
        tracker = ObjectTracker(config)

        assert tracker.next_object_id == 0
        assert len(tracker.objects) == 0

    def test_register_single_detection(self):
        """Test registering first detection."""
        config = Config()
        tracker = ObjectTracker(config)

        detection = Detection((100, 100, 200, 200), 0.9, 0, "person")
        detections = [detection]

        objects = tracker.update(detections)

        assert len(objects) == 1
        assert tracker.next_object_id == 1
        assert 0 in objects

    def test_register_multiple_detections(self):
        """Test registering multiple detections at once."""
        config = Config()
        tracker = ObjectTracker(config)

        detections = [
            Detection((100, 100, 200, 200), 0.9, 0, "person"),
            Detection((300, 300, 400, 400), 0.8, 1, "ball"),
        ]

        objects = tracker.update(detections)

        assert len(objects) == 2
        assert tracker.next_object_id == 2

    def test_update_existing_object(self):
        """Test updating an existing tracked object."""
        config = Config()
        tracker = ObjectTracker(config)

        # Frame 1: Register object
        det1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update([det1])

        # Frame 2: Update same object (moved slightly)
        det2 = Detection((110, 100, 210, 200), 0.9, 0, "person")
        objects = tracker.update([det2])

        assert len(objects) == 1
        assert tracker.next_object_id == 1  # No new object created
        obj = list(objects.values())[0]
        assert obj.age == 2
        assert obj.bbox == (110, 100, 210, 200)

    def test_disappeared_objects(self):
        """Test handling disappeared objects."""
        config = Config()
        config.max_disappeared = 3
        tracker = ObjectTracker(config)

        # Register object
        det = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update([det])

        # No detection for several frames
        tracker.update([])
        tracker.update([])

        assert len(tracker.objects) == 1
        obj = list(tracker.objects.values())[0]
        assert obj.disappeared == 2

        # One more frame - should be removed
        tracker.update([])
        assert len(tracker.objects) == 0

    def test_object_reappearance(self):
        """Test object disappearing and reappearing."""
        config = Config()
        config.max_disappeared = 5
        tracker = ObjectTracker(config)

        # Register object
        det1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update([det1])

        # Disappear for 2 frames
        tracker.update([])
        tracker.update([])

        # Reappear at similar location
        det2 = Detection((105, 100, 205, 200), 0.9, 0, "person")
        objects = tracker.update([det2])

        # Should still be same object
        assert len(objects) == 1
        assert tracker.next_object_id == 1
        obj = list(objects.values())[0]
        assert obj.disappeared == 0

    def test_new_object_after_old_removed(self):
        """Test registering new object after old one removed."""
        config = Config()
        config.max_disappeared = 2
        tracker = ObjectTracker(config)

        # Register and lose object
        det1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update([det1])
        tracker.update([])
        tracker.update([])
        tracker.update([])  # Object removed

        # New object at different location
        det2 = Detection((300, 300, 400, 400), 0.8, 1, "ball")
        objects = tracker.update([det2])

        assert len(objects) == 1
        assert tracker.next_object_id == 2

    def test_max_distance_threshold(self):
        """Test that objects too far apart aren't associated."""
        config = Config()
        config.max_distance = 50
        tracker = ObjectTracker(config)

        # Register object
        det1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update([det1])

        # Detection very far away (should be new object)
        det2 = Detection((500, 500, 600, 600), 0.9, 0, "person")
        _ = tracker.update([det2])

        # Should have 2 objects (old disappeared, new registered)
        assert tracker.next_object_id >= 2

    def test_get_primary_target(self):
        """Test getting primary tracking target."""
        config = Config()
        tracker = ObjectTracker(config)

        # No objects
        assert tracker.get_primary_target() is None

        # Add multiple objects at different times
        det1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update([det1])

        det2 = Detection((300, 300, 400, 400), 0.8, 0, "person")
        tracker.update([det1, det2])

        det3 = Detection((500, 500, 600, 600), 0.7, 0, "person")
        tracker.update([det1, det2, det3])

        # Primary should be oldest (first one)
        primary = tracker.get_primary_target()
        assert primary is not None
        assert primary.id == 0

    def test_reset(self):
        """Test resetting tracker."""
        config = Config()
        tracker = ObjectTracker(config)

        # Add objects
        detections = [
            Detection((100, 100, 200, 200), 0.9, 0, "person"),
            Detection((300, 300, 400, 400), 0.8, 1, "ball"),
        ]
        tracker.update(detections)

        assert len(tracker.objects) == 2

        # Reset
        tracker.reset()

        assert len(tracker.objects) == 0
        assert tracker.next_object_id == 0


class TestSingleObjectTracker:
    """Test SingleObjectTracker class."""

    def test_creation(self):
        """Test creating single object tracker."""
        config = Config()
        tracker = SingleObjectTracker(config)

        assert tracker.target is None
        assert not tracker.has_target()

    def test_first_detection(self):
        """Test first detection creates target."""
        config = Config()
        tracker = SingleObjectTracker(config)

        detection = Detection((100, 100, 200, 200), 0.9, 0, "person")
        target = tracker.update(detection)

        assert target is not None
        assert tracker.has_target()
        assert target.id == 0
        assert target.bbox == (100, 100, 200, 200)

    def test_update_existing_target(self):
        """Test updating existing target."""
        config = Config()
        tracker = SingleObjectTracker(config)

        det1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update(det1)

        det2 = Detection((110, 100, 210, 200), 0.85, 0, "person")
        target = tracker.update(det2)

        assert target.bbox == (110, 100, 210, 200)
        assert target.age == 2

    def test_target_lost(self):
        """Test handling lost target."""
        config = Config()
        config.max_disappeared = 3
        tracker = SingleObjectTracker(config)

        # Create target
        det = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update(det)

        # Lose target for max_disappeared frames
        tracker.update(None)
        tracker.update(None)
        tracker.update(None)

        assert tracker.target is not None
        assert tracker.target.disappeared == 3

        # One more frame - target should be removed
        tracker.update(None)
        assert tracker.target is None
        assert not tracker.has_target()

    def test_target_reacquisition(self):
        """Test reacquiring target after brief loss."""
        config = Config()
        config.max_disappeared = 5
        tracker = SingleObjectTracker(config)

        det1 = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update(det1)

        # Lose for 2 frames
        tracker.update(None)
        tracker.update(None)

        # Reacquire
        det2 = Detection((110, 100, 210, 200), 0.9, 0, "person")
        target = tracker.update(det2)

        assert target.disappeared == 0
        assert tracker.has_target()

    def test_reset(self):
        """Test resetting tracker."""
        config = Config()
        tracker = SingleObjectTracker(config)

        det = Detection((100, 100, 200, 200), 0.9, 0, "person")
        tracker.update(det)

        assert tracker.has_target()

        tracker.reset()

        assert tracker.target is None
        assert not tracker.has_target()


class TestTrackingIntegration:
    """Integration tests for complete tracking scenarios."""

    def test_single_object_continuous_tracking(self):
        """Test tracking single object across multiple frames."""
        config = Config()
        tracker = ObjectTracker(config)

        # Simulate object moving across frames
        positions = [
            (100, 100, 200, 200),
            (110, 100, 210, 200),
            (120, 100, 220, 200),
            (130, 100, 230, 200),
            (140, 100, 240, 200),
        ]

        for pos in positions:
            det = Detection(pos, 0.9, 0, "person")
            tracker.update([det])

        # Should still be single object
        assert len(tracker.objects) == 1
        assert tracker.next_object_id == 1

        obj = list(tracker.objects.values())[0]
        assert obj.age == 5
        assert len(obj.centers) == 5

    def test_multiple_objects_tracking(self):
        """Test tracking multiple objects simultaneously."""
        config = Config()
        tracker = ObjectTracker(config)

        # Two objects moving independently
        for i in range(5):
            detections = [
                Detection((100 + i * 10, 100, 200 + i * 10, 200), 0.9, 0, "person"),
                Detection((300 + i * 5, 300, 400 + i * 5, 400), 0.8, 1, "ball"),
            ]
            tracker.update(detections)

        assert len(tracker.objects) == 2

        obj1, obj2 = list(tracker.objects.values())
        assert obj1.age == 5
        assert obj2.age == 5

    def test_object_crossing_paths(self):
        """Test tracking when objects cross paths."""
        config = Config()
        tracker = ObjectTracker(config)

        # Frame 1: Two objects far apart
        det1 = Detection((100, 200, 200, 300), 0.9, 0, "person")
        det2 = Detection((400, 200, 500, 300), 0.9, 0, "person")
        tracker.update([det1, det2])

        # Frame 2: Objects closer
        det1 = Detection((200, 200, 300, 300), 0.9, 0, "person")
        det2 = Detection((300, 200, 400, 300), 0.9, 0, "person")
        tracker.update([det1, det2])

        # Should still track 2 distinct objects
        assert len(tracker.objects) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
