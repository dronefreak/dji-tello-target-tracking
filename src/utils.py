"""Utility functions for image processing, visualization, and misc operations."""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import time
from collections import deque


class FPSCounter:
    """Calculate and track FPS."""

    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)

    def update(self) -> None:
        """Update with current timestamp."""
        self.timestamps.append(time.time())

    def get_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.timestamps) < 2:
            return 0.0
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])


class PIDController:
    """PID controller for smooth drone movements."""

    def __init__(
        self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-100, 100)
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()

    def update(self, error: float) -> float:
        """Calculate PID output."""
        current_time = time.time()
        dt = current_time - self.previous_time

        if dt <= 0.0:
            dt = 1e-6

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Clamp output
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update state
        self.previous_error = error
        self.previous_time = current_time

        return output

    def reset(self) -> None:
        """Reset controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    confidence: float = 0.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box with label on frame."""
    x1, y1, x2, y2 = bbox

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label
    if label:
        label_text = f"{label}: {confidence:.2f}" if confidence > 0 else label
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)

        cv2.rectangle(
            frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y), color, -1
        )
        cv2.putText(
            frame, label_text, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return frame


def draw_crosshair(
    frame: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    size: int = 20,
    thickness: int = 2,
) -> np.ndarray:
    """Draw crosshair at frame center or specified position."""
    h, w = frame.shape[:2]

    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    # Draw crosshair
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(frame, (cx, cy), size // 2, color, thickness)

    return frame


def draw_trajectory(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw trajectory line through points."""
    if len(points) < 2:
        return frame

    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i - 1], points[i], color, thickness)

    return frame


def draw_vector(
    frame: np.ndarray,
    center: Tuple[int, int],
    target: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw directional vector from center to target."""
    cv2.arrowedLine(frame, center, target, color, thickness, tipLength=0.3)
    return frame


def draw_info_panel(
    frame: np.ndarray,
    info: dict,
    position: str = "top-left",
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.6,
) -> np.ndarray:
    """Draw semi-transparent info panel with text."""
    h, w = frame.shape[:2]
    line_height = 25
    margin = 10

    # Calculate panel size
    max_text_width = max(
        [
            cv2.getTextSize(f"{k}: {v}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            for k, v in info.items()
        ]
    )
    panel_width = max_text_width + 2 * margin
    panel_height = len(info) * line_height + 2 * margin

    # Determine position
    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = w - panel_width - margin, margin
    elif position == "bottom-left":
        x, y = margin, h - panel_height - margin
    else:  # bottom-right
        x, y = w - panel_width - margin, h - panel_height - margin

    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), bg_color, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Draw text
    text_y = y + margin + 20
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x + margin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        text_y += line_height

    return frame


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def compute_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def normalize_bbox(
    bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int
) -> Tuple[float, float, float, float]:
    """Normalize bounding box coordinates to [0, 1]."""
    x1, y1, x2, y2 = bbox
    return (x1 / frame_width, y1 / frame_height, x2 / frame_width, y2 / frame_height)


def denormalize_bbox(
    bbox: Tuple[float, float, float, float], frame_width: int, frame_height: int
) -> Tuple[int, int, int, int]:
    """Convert normalized bbox back to pixel coordinates."""
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * frame_width),
        int(y1 * frame_height),
        int(x2 * frame_width),
        int(y2 * frame_height),
    )


def resize_with_aspect_ratio(
    frame: np.ndarray, target_width: Optional[int] = None, target_height: Optional[int] = None
) -> np.ndarray:
    """Resize frame while maintaining aspect ratio."""
    h, w = frame.shape[:2]

    if target_width is None and target_height is None:
        return frame

    if target_width is not None:
        ratio = target_width / w
        new_size = (target_width, int(h * ratio))
    else:
        ratio = target_height / h
        new_size = (int(w * ratio), target_height)

    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def create_color_mask(
    frame: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]
) -> np.ndarray:
    """Create HSV color mask for tracking."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def find_largest_contour(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find largest contour in mask and return its bounding box."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 100:  # Min area threshold
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, x + w, y + h)
