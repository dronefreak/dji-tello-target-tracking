"""Unit tests for utility functions.

Tests PID controller, FPS counter, drawing functions, and image processing.
"""

import pytest
import numpy as np
import cv2
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    FPSCounter,
    PIDController,
    draw_bbox,
    draw_crosshair,
    draw_trajectory,
    draw_vector,
    draw_info_panel,
    compute_iou,
    get_bbox_center,
    compute_distance,
    normalize_bbox,
    denormalize_bbox,
    resize_with_aspect_ratio,
    create_color_mask,
    find_largest_contour,
)


class TestFPSCounter:
    """Test FPSCounter class."""

    def test_creation(self):
        """Test creating FPS counter."""
        fps = FPSCounter(window_size=10)
        assert len(fps.timestamps) == 0

    def test_initial_fps(self):
        """Test FPS before any updates."""
        fps = FPSCounter()
        assert fps.get_fps() == 0.0

    def test_single_update(self):
        """Test FPS with single update."""
        fps = FPSCounter()
        fps.update()
        assert fps.get_fps() == 0.0  # Need at least 2 timestamps

    def test_fps_calculation(self):
        """Test FPS calculation with multiple updates."""
        fps = FPSCounter()

        # Simulate 10 frames at ~30 FPS (33ms per frame)
        for _ in range(10):
            fps.update()
            time.sleep(0.033)

        calculated_fps = fps.get_fps()

        # Should be around 30 FPS (allow some tolerance)
        assert 25 < calculated_fps < 35

    def test_window_size_limit(self):
        """Test that window size is respected."""
        fps = FPSCounter(window_size=5)

        # Add more updates than window size
        for _ in range(10):
            fps.update()

        assert len(fps.timestamps) == 5


class TestPIDController:
    """Test PIDController class."""

    def test_creation(self):
        """Test creating PID controller."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.05)

        assert pid.kp == 1.0
        assert pid.ki == 0.1
        assert pid.kd == 0.05
        assert pid.integral == 0.0
        assert pid.previous_error == 0.0

    def test_proportional_only(self):
        """Test proportional control only."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)

        error = 10.0
        output = pid.update(error)

        # With only P, output should equal error
        assert abs(output - error) < 0.1

    def test_output_limits(self):
        """Test output clamping."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0, output_limits=(-50, 50))

        error = 100.0
        output = pid.update(error)

        # Should be clamped to 50
        assert output == 50.0

        error = -100.0
        output = pid.update(error)

        # Should be clamped to -50
        assert output == -50.0

    def test_integral_windup_prevention(self):
        """Test that output limits prevent integral windup."""
        pid = PIDController(kp=0.5, ki=0.5, kd=0.0, output_limits=(-100, 100))

        # Apply large error for multiple steps
        for _ in range(10):
            pid.update(1000.0)
            time.sleep(0.01)

        # Output should still be limited
        output = pid.update(1000.0)
        assert output == 100.0

    def test_derivative_smoothing(self):
        """Test derivative term responds to error changes."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)

        # Constant error - derivative should be ~0
        pid.update(10.0)
        time.sleep(0.1)
        output1 = pid.update(10.0)
        assert abs(output1) < 1.0

        # Sudden change - derivative should respond
        time.sleep(0.1)
        output2 = pid.update(20.0)
        assert abs(output2) > 1.0

    def test_reset(self):
        """Test resetting PID controller."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)

        # Run for a bit
        for _ in range(5):
            pid.update(10.0)
            time.sleep(0.01)

        assert pid.integral != 0.0

        # Reset
        pid.reset()

        assert pid.integral == 0.0
        assert pid.previous_error == 0.0


class TestDrawingFunctions:
    """Test drawing utility functions."""

    def test_draw_bbox(self):
        """Test drawing bounding box."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)

        result = draw_bbox(frame, bbox, "test", 0.95, (0, 255, 0))

        assert result.shape == frame.shape
        # Check that something was drawn (frame changed)
        assert not np.array_equal(result, frame)

    def test_draw_crosshair_center(self):
        """Test drawing crosshair at center."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_crosshair(frame)

        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_draw_crosshair_custom_position(self):
        """Test drawing crosshair at custom position."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_crosshair(frame, center=(100, 100))

        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_draw_trajectory(self):
        """Test drawing trajectory."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        points = [(100, 100), (150, 150), (200, 200), (250, 250)]

        result = draw_trajectory(frame, points)

        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_draw_trajectory_empty(self):
        """Test drawing trajectory with no points."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_trajectory(frame, [])

        # Should return unchanged
        assert np.array_equal(result, frame)

    def test_draw_vector(self):
        """Test drawing directional vector."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_vector(frame, (100, 100), (200, 200))

        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_draw_info_panel(self):
        """Test drawing info panel."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = {"FPS": "30.0", "Target": "person", "Confidence": "0.95"}

        result = draw_info_panel(frame, info)

        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_draw_info_panel_positions(self):
        """Test info panel at different positions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        info = {"Test": "Value"}

        positions = ["top-left", "top-right", "bottom-left", "bottom-right"]

        for pos in positions:
            result = draw_info_panel(frame, info, position=pos)
            assert result.shape == frame.shape


class TestGeometryFunctions:
    """Test geometry and bbox utility functions."""

    def test_compute_iou_no_overlap(self):
        """Test IoU with no overlap."""
        box1 = (0, 0, 100, 100)
        box2 = (200, 200, 300, 300)

        iou = compute_iou(box1, box2)

        assert iou == 0.0

    def test_compute_iou_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        box1 = (100, 100, 200, 200)
        box2 = (100, 100, 200, 200)

        iou = compute_iou(box1, box2)

        assert iou == 1.0

    def test_compute_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)

        iou = compute_iou(box1, box2)

        # Should be between 0 and 1
        assert 0 < iou < 1
        # Known value for this case is ~0.14
        assert 0.1 < iou < 0.2

    def test_get_bbox_center(self):
        """Test bbox center calculation."""
        test_cases = [
            ((0, 0, 100, 100), (50, 50)),
            ((100, 100, 200, 200), (150, 150)),
            ((50, 75, 150, 175), (100, 125)),
        ]

        for bbox, expected in test_cases:
            center = get_bbox_center(bbox)
            assert center == expected

    def test_compute_distance(self):
        """Test distance calculation."""
        # Horizontal distance
        dist1 = compute_distance((0, 0), (3, 0))
        assert dist1 == 3.0

        # Vertical distance
        dist2 = compute_distance((0, 0), (0, 4))
        assert dist2 == 4.0

        # Diagonal distance (3-4-5 triangle)
        dist3 = compute_distance((0, 0), (3, 4))
        assert dist3 == 5.0

    def test_normalize_bbox(self):
        """Test bbox normalization."""
        bbox = (100, 100, 200, 200)
        width, height = 640, 480

        normalized = normalize_bbox(bbox, width, height)

        # Check values are in [0, 1]
        for val in normalized:
            assert 0 <= val <= 1

        # Check specific values
        assert abs(normalized[0] - 100 / 640) < 0.001
        assert abs(normalized[1] - 100 / 480) < 0.001

    def test_denormalize_bbox(self):
        """Test bbox denormalization."""
        normalized = (0.15625, 0.20833, 0.3125, 0.41666)
        width, height = 640, 480

        bbox = denormalize_bbox(normalized, width, height)

        # Should be approximately (100, 100, 200, 200)
        assert 95 <= bbox[0] <= 105
        assert 95 <= bbox[1] <= 105
        assert 195 <= bbox[2] <= 205
        assert 195 <= bbox[3] <= 205

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize/denormalize are inverse operations."""
        original = (100, 100, 200, 200)
        width, height = 640, 480

        normalized = normalize_bbox(original, width, height)
        recovered = denormalize_bbox(normalized, width, height)

        # Should be approximately equal
        for orig, rec in zip(original, recovered):
            assert abs(orig - rec) < 1


class TestImageProcessing:
    """Test image processing functions."""

    def test_resize_with_aspect_ratio_width(self):
        """Test resizing by target width."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = resize_with_aspect_ratio(frame, target_width=320)

        assert result.shape[1] == 320
        assert result.shape[0] == 240  # Maintains aspect ratio

    def test_resize_with_aspect_ratio_height(self):
        """Test resizing by target height."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = resize_with_aspect_ratio(frame, target_height=240)

        assert result.shape[0] == 240
        assert result.shape[1] == 320  # Maintains aspect ratio

    def test_resize_no_params(self):
        """Test resize with no parameters returns original."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = resize_with_aspect_ratio(frame)

        assert np.array_equal(result, frame)

    def test_create_color_mask_green(self):
        """Test creating green color mask."""
        # Create frame with green rectangle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), -1)

        lower = (50, 50, 50)
        upper = (70, 255, 255)

        mask = create_color_mask(frame, lower, upper)

        # Mask should have white pixels in green area
        assert mask.shape == (480, 640)
        assert mask[300, 300] > 0  # Center of rectangle
        assert mask[100, 100] == 0  # Outside rectangle

    def test_find_largest_contour(self):
        """Test finding largest contour."""
        # Create mask with two white rectangles (different sizes)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)  # Small
        cv2.rectangle(mask, (300, 300), (500, 500), 255, -1)  # Large

        bbox = find_largest_contour(mask)

        assert bbox is not None

        # Should be the larger rectangle
        x1, y1, x2, y2 = bbox
        assert x1 > 250
        assert y1 > 250

    def test_find_largest_contour_empty_mask(self):
        """Test finding contour in empty mask."""
        mask = np.zeros((480, 640), dtype=np.uint8)

        bbox = find_largest_contour(mask)

        assert bbox is None

    def test_find_largest_contour_small_filtered(self):
        """Test that very small contours are filtered."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 3, 255, -1)  # Tiny circle

        bbox = find_largest_contour(mask)

        # Should be filtered out (area < 100)
        assert bbox is None


class TestIntegration:
    """Integration tests for utils."""

    def test_pid_tracking_simulation(self):
        """Test PID controller tracking a moving target."""
        pid = PIDController(kp=0.5, ki=0.1, kd=0.2, output_limits=(-100, 100))

        # Simulate tracking a target moving from position 0 to 100
        position = 0.0
        target = 100.0

        outputs = []

        # Run for 20 steps
        for _ in range(20):
            error = target - position
            output = pid.update(error)
            outputs.append(output)

            # Simulate movement based on output
            position += output * 0.1
            time.sleep(0.01)

        # Position should have moved toward target
        assert position > 50  # At least halfway

        # Output should decrease as we approach target
        assert abs(outputs[-1]) < abs(outputs[0])

    def test_full_drawing_pipeline(self):
        """Test combining multiple drawing functions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw bbox
        frame = draw_bbox(frame, (100, 100, 200, 200), "test", 0.95)

        # Draw trajectory
        points = [(150, 150), (160, 160), (170, 170)]
        frame = draw_trajectory(frame, points)

        # Draw crosshair
        frame = draw_crosshair(frame)

        # Draw info panel
        info = {"Test": "Value"}
        frame = draw_info_panel(frame, info)

        # Frame should have been modified
        assert not np.all(frame == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
