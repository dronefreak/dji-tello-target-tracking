"""Unit tests for the detector module.

Tests detection, HSV color tracking, and hybrid modes.
"""

import pytest
import numpy as np
import cv2

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.detector import Detection, ObjectDetector, HSVDetector, HybridDetector

# Skip tests if ultralytics not available
try:
    from ultralytics import YOLO  # F401

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class TestDetection:
    """Test Detection class."""

    def test_creation(self):
        """Test creating a detection."""
        det = Detection(bbox=(100, 100, 200, 200), confidence=0.95, class_id=0, class_name="person")

        assert det.bbox == (100, 100, 200, 200)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.center == (150, 150)

    def test_center_calculation(self):
        """Test center point calculation."""
        det = Detection((50, 50, 150, 150), 0.9, 0, "person")
        assert det.center == (100, 100)

        det2 = Detection((0, 0, 100, 200), 0.9, 0, "person")
        assert det2.center == (50, 100)

    def test_repr(self):
        """Test string representation."""
        det = Detection((100, 100, 200, 200), 0.95, 0, "person")
        repr_str = repr(det)

        assert "Detection" in repr_str
        assert "person" in repr_str
        assert "0.95" in repr_str


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
class TestObjectDetector:
    """Test ObjectDetector class."""

    def test_creation(self):
        """Test creating detector."""
        config = Config()
        detector = ObjectDetector(config)

        assert detector.model is not None
        assert detector.device == config.device

    def test_detect_on_blank_frame(self):
        """Test detection on blank frame."""
        config = Config()
        detector = ObjectDetector(config)

        # Create blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        # Should return empty list or very low confidence
        assert isinstance(detections, list)

    def test_detect_returns_list(self):
        """Test that detect returns a list."""
        config = Config()
        detector = ObjectDetector(config)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        assert isinstance(detections, list)

    def test_detect_best(self):
        """Test detect_best returns single detection or None."""
        config = Config()
        detector = ObjectDetector(config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        best = detector.detect_best(frame)

        # Should be None or Detection
        assert best is None or isinstance(best, Detection)

    def test_detect_closest_to_center(self):
        """Test detect_closest_to_center."""
        config = Config()
        detector = ObjectDetector(config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        closest = detector.detect_closest_to_center(frame)

        assert closest is None or isinstance(closest, Detection)

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        config = Config()
        config.confidence_threshold = 0.9  # Very high threshold

        detector = ObjectDetector(config)

        # Random frame unlikely to have high confidence
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        # All detections should meet threshold
        for det in detections:
            assert det.confidence >= config.confidence_threshold

    def test_target_classes_filtering(self):
        """Test filtering by target classes."""
        config = Config()
        config.target_classes = ["person"]  # Only detect persons

        detector = ObjectDetector(config)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        # All detections should be person class
        for det in detections:
            assert det.class_name == "person"

    def test_get_model_info(self):
        """Test getting model information."""
        config = Config()
        detector = ObjectDetector(config)

        info = detector.get_model_info()

        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "num_classes" in info


class TestHSVDetector:
    """Test HSVDetector class."""

    def test_creation(self):
        """Test creating HSV detector."""
        config = Config()
        detector = HSVDetector(config, "green")

        assert detector.color == "green"
        assert detector.lower == config.hsv_ranges["green"][0]
        assert detector.upper == config.hsv_ranges["green"][1]

    def test_invalid_color(self):
        """Test creating detector with invalid color."""
        config = Config()

        with pytest.raises(ValueError):
            HSVDetector(config, "invalid_color")

    def test_detect_on_blank_frame(self):
        """Test detection on blank frame."""
        config = Config()
        detector = HSVDetector(config, "green")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = detector.detect(frame)

        # No green in blank frame
        assert detection is None

    def test_detect_green_object(self):
        """Test detecting green object."""
        config = Config()
        detector = HSVDetector(config, "green")

        # Create frame with green rectangle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), -1)

        detection = detector.detect(frame)

        assert detection is not None
        assert isinstance(detection, Detection)
        assert detection.class_name == "green"
        assert detection.confidence == 1.0

        # Check bbox is reasonable
        x1, y1, x2, y2 = detection.bbox
        assert 180 < x1 < 220
        assert 180 < y1 < 220
        assert 380 < x2 < 420
        assert 380 < y2 < 420

    def test_detect_red_object(self):
        """Test detecting red object."""
        config = Config()
        detector = HSVDetector(config, "red")

        # Create frame with red rectangle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (400, 400), (0, 0, 255), -1)

        detection = detector.detect(frame)

        assert detection is not None
        assert detection.class_name == "red"

    def test_detect_blue_object(self):
        """Test detecting blue object."""
        config = Config()
        detector = HSVDetector(config, "blue")

        # Create frame with blue rectangle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (400, 400), (255, 0, 0), -1)

        detection = detector.detect(frame)

        assert detection is not None
        assert detection.class_name == "blue"

    def test_detect_small_object_filtered(self):
        """Test that very small objects are filtered out."""
        config = Config()
        detector = HSVDetector(config, "green")

        # Create frame with tiny green dot
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 3, (0, 255, 0), -1)

        detection = detector.detect(frame)

        # Should be filtered out (area < 100)
        assert detection is None

    def test_detect_largest_contour(self):
        """Test that largest contour is selected."""
        config = Config()
        detector = HSVDetector(config, "green")

        # Create frame with two green rectangles (different sizes)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), -1)  # Small
        cv2.rectangle(frame, (300, 300), (500, 500), (0, 255, 0), -1)  # Large

        detection = detector.detect(frame)

        assert detection is not None

        # Should detect the larger rectangle
        x1, y1, x2, y2 = detection.bbox
        assert x1 > 250  # Should be the right rectangle


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
class TestHybridDetector:
    """Test HybridDetector class."""

    def test_creation_yolo_mode(self):
        """Test creating hybrid detector in YOLO mode."""
        config = Config()
        detector = HybridDetector(config, use_yolo=True)

        assert detector.use_yolo is True
        assert isinstance(detector.detector, ObjectDetector)

    def test_creation_hsv_mode(self):
        """Test creating hybrid detector in HSV mode."""
        config = Config()
        detector = HybridDetector(config, use_yolo=False, hsv_color="green")

        assert detector.use_yolo is False
        assert isinstance(detector.detector, HSVDetector)

    def test_detect_yolo_mode(self):
        """Test detection in YOLO mode."""
        config = Config()
        detector = HybridDetector(config, use_yolo=True)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detection = detector.detect(frame)

        assert detection is None or isinstance(detection, Detection)

    def test_detect_hsv_mode(self):
        """Test detection in HSV mode."""
        config = Config()
        detector = HybridDetector(config, use_yolo=False, hsv_color="green")

        # Frame with green object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), -1)

        detection = detector.detect(frame)

        assert detection is not None
        assert detection.class_name == "green"


class TestDetectionHelpers:
    """Test detection helper functions."""

    def test_detection_center_calculation(self):
        """Test center calculation for various bboxes."""
        test_cases = [
            ((0, 0, 100, 100), (50, 50)),
            ((100, 100, 200, 200), (150, 150)),
            ((50, 75, 150, 175), (100, 125)),
        ]

        for bbox, expected_center in test_cases:
            det = Detection(bbox, 0.9, 0, "test")
            assert det.center == expected_center


class TestIntegration:
    """Integration tests for detector."""

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
    def test_yolo_on_test_image(self):
        """Test YOLO detector on a test image."""
        config = Config()
        config.confidence_threshold = 0.3  # Lower for test
        detector = ObjectDetector(config)

        # Create simple test image with shapes
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add some colored rectangles
        cv2.rectangle(frame, (100, 100), (200, 200), (0, 0, 255), -1)
        cv2.rectangle(frame, (300, 300), (400, 400), (0, 255, 0), -1)
        cv2.circle(frame, (500, 100), 50, (255, 0, 0), -1)

        detections = detector.detect(frame)

        # May or may not detect anything in synthetic image
        assert isinstance(detections, list)

    def test_hsv_full_pipeline(self):
        """Test HSV detector full pipeline."""
        config = Config()
        detector = HSVDetector(config, "green")

        # Frame 1: Green ball on left
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame1, (150, 240), 50, (0, 255, 0), -1)

        det1 = detector.detect(frame1)
        assert det1 is not None

        # Frame 2: Green ball moved right
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame2, (250, 240), 50, (0, 255, 0), -1)

        det2 = detector.detect(frame2)
        assert det2 is not None

        # Center should have moved right
        assert det2.center[0] > det1.center[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
