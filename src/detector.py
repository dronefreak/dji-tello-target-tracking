"""PyTorch-based object detection using YOLOv8 via ultralytics.

Modern replacement for the old HSV color tracking.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

from src.config import Config
from src.utils import get_bbox_center


class Detection:
    """Single detection result."""

    def __init__(
        self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int, class_name: str
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.center = get_bbox_center(bbox)

    def __repr__(self) -> str:
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


class ObjectDetector:
    """Modern object detector using YOLOv8.

    Supports various YOLO models and can filter by target classes.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.device = config.device

        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. " "Install with: pip install ultralytics")

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8 model."""
        print(f"Loading {self.config.model_name} on {self.device}...")

        try:
            self.model = YOLO(f"{self.config.model_name}.pt")
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []

        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )[0]

        detections = []

        # Parse results
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[cls_id]

                # Filter by target classes if specified
                if self.config.target_classes and class_name not in self.config.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=class_name,
                )
                detections.append(detection)

        return detections

    def detect_best(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect and return only the best (highest confidence) detection.

        Useful for single-target tracking.
        """
        detections = self.detect(frame)

        if not detections:
            return None

        return max(detections, key=lambda d: d.confidence)

    def detect_closest_to_center(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect and return the detection closest to frame center.

        Useful for drone tracking where target is typically centered.
        """
        detections = self.detect(frame)

        if not detections:
            return None

        h, w = frame.shape[:2]
        frame_center = (w // 2, h // 2)

        # Find detection with center closest to frame center
        closest = min(
            detections,
            key=lambda d: np.sqrt(
                (d.center[0] - frame_center[0]) ** 2 + (d.center[1] - frame_center[1]) ** 2
            ),
        )

        return closest

    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded model."""
        if self.model is None:
            return {}

        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "class_names": list(self.model.names.values()),
            "num_classes": len(self.model.names),
        }


class HSVDetector:
    """Fallback HSV color-based detector.

    Kept for backwards compatibility and situations where YOLO is too heavy.
    """

    def __init__(self, config: Config, color: str = "green"):
        self.config = config
        self.color = color

        if color not in config.hsv_ranges:
            raise ValueError(
                f"Color {color} not in config. Available: {list(config.hsv_ranges.keys())}"
            )

        self.lower, self.upper = config.hsv_ranges[color]

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect object using HSV color masking.

        Returns single detection (largest contour).
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 100:
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        bbox = (x, y, x + w, y + h)

        # Return as Detection object for consistency
        return Detection(
            bbox=bbox,
            confidence=1.0,  # HSV doesn't have confidence
            class_id=0,
            class_name=self.color,
        )


class HybridDetector:
    """Hybrid detector that can switch between YOLO and HSV.

    Useful for testing or fallback scenarios.
    """

    def __init__(self, config: Config, use_yolo: bool = True, hsv_color: str = "green"):
        self.config = config
        self.use_yolo = use_yolo and YOLO_AVAILABLE

        if self.use_yolo:
            self.detector = ObjectDetector(config)
            print("Using YOLO detector")
        else:
            self.detector = HSVDetector(config, hsv_color)
            print(f"Using HSV detector for {hsv_color}")

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect using current detector."""
        if self.use_yolo:
            return self.detector.detect_closest_to_center(frame)
        else:
            return self.detector.detect(frame)

    def switch_mode(self) -> None:
        """Switch between YOLO and HSV detection."""
        if not YOLO_AVAILABLE:
            print("Cannot switch to YOLO - ultralytics not installed")
            return

        self.use_yolo = not self.use_yolo
        print(f"Switched to {'YOLO' if self.use_yolo else 'HSV'} detector")
