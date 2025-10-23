"""Object tracking with centroid association and trajectory history.

Maintains object IDs across frames using distance-based matching.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict, deque

from src.config import Config
from src.detector import Detection
from src.utils import compute_distance


class TrackedObject:
    """Represents a tracked object with history."""

    def __init__(self, object_id: int, detection: Detection, max_history: int = 30):
        self.id = object_id
        self.class_name = detection.class_name
        self.class_id = detection.class_id

        # Current state
        self.bbox = detection.bbox
        self.center = detection.center
        self.confidence = detection.confidence

        # History
        self.max_history = max_history
        self.centers = deque([self.center], maxlen=max_history)
        self.bboxes = deque([self.bbox], maxlen=max_history)

        # Tracking info
        self.disappeared = 0
        self.age = 1  # Frames since creation

    def update(self, detection: Detection) -> None:
        """Update tracked object with new detection."""
        self.bbox = detection.bbox
        self.center = detection.center
        self.confidence = detection.confidence

        self.centers.append(self.center)
        self.bboxes.append(self.bbox)

        self.disappeared = 0
        self.age += 1

    def mark_disappeared(self) -> None:
        """Increment disappeared counter."""
        self.disappeared += 1

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Calculate velocity vector from recent trajectory."""
        if len(self.centers) < 2:
            return None

        # Use last 5 frames for velocity estimation
        recent = list(self.centers)[-5:]
        if len(recent) < 2:
            return None

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]

        return (dx / len(recent), dy / len(recent))

    def predict_next_position(self) -> Optional[Tuple[int, int]]:
        """Predict next position based on velocity."""
        velocity = self.get_velocity()
        if velocity is None:
            return self.center

        vx, vy = velocity
        predicted_x = int(self.center[0] + vx)
        predicted_y = int(self.center[1] + vy)

        return (predicted_x, predicted_y)

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self.id}, class={self.class_name}, "
            f"center={self.center}, age={self.age}, disappeared={self.disappeared})"
        )


class ObjectTracker:
    """Centroid-based object tracker.

    Associates detections across frames using distance metrics.
    """

    def __init__(self, config: Config):
        self.config = config
        self.next_object_id = 0
        self.objects: OrderedDict[int, TrackedObject] = OrderedDict()

        self.max_disappeared = config.max_disappeared
        self.max_distance = config.max_distance

    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """Update tracker with new detections.

        Args:
            detections: List of Detection objects from current frame

        Returns:
            Dictionary of tracked objects {object_id: TrackedObject}
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].mark_disappeared()

                # Remove objects that disappeared too long
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    self.deregister(obj_id)

            return self.objects

        # Extract centers from detections
        detection_centers = [det.center for det in detections]

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)

        else:
            # Match existing objects to detections
            object_ids = list(self.objects.keys())
            object_centers = [self.objects[obj_id].center for obj_id in object_ids]

            # Compute distance matrix
            distances = np.zeros((len(object_centers), len(detection_centers)))
            for i, obj_center in enumerate(object_centers):
                for j, det_center in enumerate(detection_centers):
                    distances[i, j] = compute_distance(obj_center, det_center)

            # Match using Hungarian algorithm (greedy approximation)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Update matched objects
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Check if distance is acceptable
                if distances[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id].update(detections[col])

                used_rows.add(row)
                used_cols.add(col)

            # Mark unmatched objects as disappeared
            unused_rows = set(range(len(object_centers))) - used_rows
            for row in unused_rows:
                obj_id = object_ids[row]
                self.objects[obj_id].mark_disappeared()

                if self.objects[obj_id].disappeared > self.max_disappeared:
                    self.deregister(obj_id)

            # Register new detections
            unused_cols = set(range(len(detection_centers))) - used_cols
            for col in unused_cols:
                self.register(detections[col])

        return self.objects

    def register(self, detection: Detection) -> int:
        """Register a new tracked object."""
        obj_id = self.next_object_id
        self.objects[obj_id] = TrackedObject(obj_id, detection)
        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id: int) -> None:
        """Remove an object from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]

    def get_object(self, object_id: int) -> Optional[TrackedObject]:
        """Get tracked object by ID."""
        return self.objects.get(object_id)

    def get_all_objects(self) -> List[TrackedObject]:
        """Get all currently tracked objects."""
        return list(self.objects.values())

    def get_primary_target(self) -> Optional[TrackedObject]:
        """Get the primary tracking target.

        Uses heuristics: oldest object with lowest disappeared count.
        """
        if not self.objects:
            return None

        # Filter out objects that are disappearing
        active_objects = [obj for obj in self.objects.values() if obj.disappeared == 0]

        if not active_objects:
            return None

        # Return oldest active object (most stable)
        return max(active_objects, key=lambda obj: obj.age)

    def reset(self) -> None:
        """Reset tracker state."""
        self.objects.clear()
        self.next_object_id = 0


class SingleObjectTracker:
    """Simplified tracker for single-target tracking.

    Useful for drone tracking where we follow one object.
    """

    def __init__(self, config: Config):
        self.config = config
        self.target: Optional[TrackedObject] = None
        self.max_disappeared = config.max_disappeared

    def update(self, detection: Optional[Detection]) -> Optional[TrackedObject]:
        """Update with single detection.

        Args:
            detection: Single Detection object or None

        Returns:
            Current tracked target or None
        """
        if detection is None:
            if self.target is not None:
                self.target.mark_disappeared()

                if self.target.disappeared > self.max_disappeared:
                    self.target = None

            return self.target

        # If no target, create new one
        if self.target is None:
            self.target = TrackedObject(0, detection)
        else:
            # Update existing target
            self.target.update(detection)

        return self.target

    def reset(self) -> None:
        """Reset tracker."""
        self.target = None

    def has_target(self) -> bool:
        """Check if currently tracking a target."""
        return self.target is not None and self.target.disappeared == 0


class TrackingVisualizer:
    """Helper class for visualizing tracking results."""

    @staticmethod
    def draw_tracked_objects(
        frame: np.ndarray,
        objects: List[TrackedObject],
        show_trajectory: bool = True,
        show_velocity: bool = True,
    ) -> np.ndarray:
        """Draw all tracked objects on frame."""
        import cv2
        from src.utils import draw_bbox, draw_trajectory, draw_vector

        for obj in objects:
            # Determine color based on state
            if obj.disappeared > 0:
                color = (0, 165, 255)  # Orange for disappearing
            else:
                color = (0, 255, 0)  # Green for active

            # Draw bounding box
            label = f"ID:{obj.id} {obj.class_name}"
            frame = draw_bbox(frame, obj.bbox, label, obj.confidence, color)

            # Draw trajectory
            if show_trajectory and len(obj.centers) > 1:
                frame = draw_trajectory(frame, list(obj.centers), color)

            # Draw velocity vector
            if show_velocity:
                velocity = obj.get_velocity()
                if velocity is not None:
                    vx, vy = velocity
                    end_point = (int(obj.center[0] + vx * 10), int(obj.center[1] + vy * 10))
                    frame = draw_vector(frame, obj.center, end_point, (255, 0, 0))

        return frame
