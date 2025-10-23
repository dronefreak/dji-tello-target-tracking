"""Unit tests for configuration module.

Tests Config, DroneConfig, and ConfigBuilder classes.
"""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Config,
    DroneConfig,
    ConfigBuilder,
    get_default_config,
    get_webcam_config,
    get_drone_config,
)


class TestConfig:
    """Test Config class."""

    def test_default_creation(self):
        """Test creating config with defaults."""
        config = Config()

        assert config.model_name == "yolov8n"
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert isinstance(config.target_classes, list)
        assert config.drone_speed == 50
        assert config.frame_width == 960
        assert config.frame_height == 720

    def test_device_selection(self):
        """Test device selection (cuda/cpu)."""
        config = Config()

        # Should be cuda or cpu depending on availability
        assert config.device in ["cuda", "cpu"]

        # Should match torch availability
        if torch.cuda.is_available():
            assert config.device == "cuda"
        else:
            assert config.device == "cpu"

    def test_hsv_ranges(self):
        """Test HSV color ranges."""
        config = Config()

        assert "green" in config.hsv_ranges
        assert "red" in config.hsv_ranges
        assert "blue" in config.hsv_ranges

        # Check format (lower, upper) tuples
        green_lower, green_upper = config.hsv_ranges["green"]
        assert len(green_lower) == 3
        assert len(green_upper) == 3

    def test_pid_parameters(self):
        """Test PID controller parameters."""
        config = Config()

        assert len(config.pid_x) == 3  # Kp, Ki, Kd
        assert len(config.pid_y) == 3
        assert len(config.pid_z) == 3
        assert len(config.pid_yaw) == 3

        # All should be tuples of floats
        for pid in [config.pid_x, config.pid_y, config.pid_z, config.pid_yaw]:
            assert all(isinstance(val, (int, float)) for val in pid)

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = Config()
        config.model_name = "yolov8s"
        config.confidence_threshold = 0.7
        config.frame_width = 1280

        assert config.model_name == "yolov8s"
        assert config.confidence_threshold == 0.7
        assert config.frame_width == 1280


class TestDroneConfig:
    """Test DroneConfig class."""

    def test_default_creation(self):
        """Test creating drone config with defaults."""
        config = DroneConfig()

        assert config.tello_ip == "192.168.10.1"
        assert config.tello_port == 8889
        assert config.max_speed == 100
        assert config.min_battery == 10
        assert config.takeoff_height == 80

    def test_safety_limits(self):
        """Test safety limit parameters."""
        config = DroneConfig()

        assert config.min_battery >= 5  # Reasonable minimum
        assert config.max_speed <= 100  # Tello max
        assert config.max_tilt <= 90  # Physical limit

    def test_custom_values(self):
        """Test custom drone config values."""
        config = DroneConfig()
        config.min_battery = 20
        config.max_speed = 50

        assert config.min_battery == 20
        assert config.max_speed == 50


class TestConfigBuilder:
    """Test ConfigBuilder class."""

    def test_basic_building(self):
        """Test building config with builder pattern."""
        config = ConfigBuilder().build()

        assert isinstance(config, Config)

    def test_with_model(self):
        """Test setting model via builder."""
        config = ConfigBuilder().with_model("yolov8m").build()

        assert config.model_name == "yolov8m"

    def test_with_confidence(self):
        """Test setting confidence via builder."""
        config = ConfigBuilder().with_confidence(0.7).build()

        assert config.confidence_threshold == 0.7

    def test_with_device(self):
        """Test setting device via builder."""
        config = ConfigBuilder().with_device("cpu").build()

        assert config.device == "cpu"

    def test_with_target_classes(self):
        """Test setting target classes via builder."""
        classes = ["person", "car"]
        config = ConfigBuilder().with_target_classes(classes).build()

        assert config.target_classes == classes

    def test_with_drone_speed(self):
        """Test setting drone speed via builder."""
        config = ConfigBuilder().with_drone_speed(75).build()

        assert config.drone_speed == 75

    def test_chaining(self):
        """Test chaining multiple builder methods."""
        config = (
            ConfigBuilder()
            .with_model("yolov8s")
            .with_confidence(0.6)
            .with_device("cpu")
            .with_drone_speed(60)
            .build()
        )

        assert config.model_name == "yolov8s"
        assert config.confidence_threshold == 0.6
        assert config.device == "cpu"
        assert config.drone_speed == 60


class TestConfigFactories:
    """Test configuration factory functions."""

    def test_get_default_config(self):
        """Test getting default config."""
        config = get_default_config()

        assert isinstance(config, Config)
        assert config.model_name == "yolov8n"

    def test_get_webcam_config(self):
        """Test getting webcam-optimized config."""
        config = get_webcam_config()

        assert isinstance(config, Config)
        assert config.model_name == "yolov8n"  # Fast model
        assert config.display_fps is True
        assert config.display_tracking_info is True

    def test_get_drone_config(self):
        """Test getting drone-optimized config."""
        config, drone_config = get_drone_config()

        assert isinstance(config, Config)
        assert isinstance(drone_config, DroneConfig)

        # Drone config should use better model
        assert config.model_name in ["yolov8s", "yolov8m"]
        assert config.confidence_threshold >= 0.5

    def test_webcam_vs_drone_config_differences(self):
        """Test that webcam and drone configs are different."""
        webcam = get_webcam_config()
        drone, _ = get_drone_config()

        # Drone should use more accurate model
        assert webcam.model_name == "yolov8n"
        assert drone.model_name != "yolov8n"

        # Drone should have higher confidence threshold
        assert drone.confidence_threshold >= webcam.confidence_threshold


class TestConfigValidation:
    """Test config validation and edge cases."""

    def test_confidence_range(self):
        """Test confidence threshold is in valid range."""
        config = Config()

        # Default should be valid
        assert 0.0 <= config.confidence_threshold <= 1.0

        # Test setting extreme values
        config.confidence_threshold = 0.1
        assert config.confidence_threshold == 0.1

        config.confidence_threshold = 0.9
        assert config.confidence_threshold == 0.9

    def test_speed_limits(self):
        """Test drone speed limits."""
        config = Config()

        # Default should be reasonable
        assert 0 < config.drone_speed <= 100

        drone_config = DroneConfig()
        assert 0 < drone_config.max_speed <= 100

    def test_frame_dimensions(self):
        """Test frame dimension values."""
        config = Config()

        # Should be positive
        assert config.frame_width > 0
        assert config.frame_height > 0

        # Should be reasonable video dimensions
        assert config.frame_width >= 320
        assert config.frame_height >= 240

    def test_pid_parameters_validity(self):
        """Test PID parameters are valid."""
        config = Config()

        for pid_params in [config.pid_x, config.pid_y, config.pid_z, config.pid_yaw]:
            kp, ki, kd = pid_params

            # All should be non-negative
            assert kp >= 0
            assert ki >= 0
            assert kd >= 0


class TestConfigIntegration:
    """Integration tests for config system."""

    def test_config_for_webcam_demo(self):
        """Test config suitable for webcam demo."""
        config = get_webcam_config()

        # Should be optimized for speed
        assert config.model_name == "yolov8n"

        # Should have display features enabled
        assert config.display_fps
        assert config.display_tracking_info

        # Reasonable confidence for demo
        assert 0.3 <= config.confidence_threshold <= 0.6

    def test_config_for_drone_tracking(self):
        """Test config suitable for drone tracking."""
        config, drone_config = get_drone_config()

        # Should prioritize accuracy over speed
        assert config.model_name in ["yolov8s", "yolov8m", "yolov8l"]

        # Higher confidence for safety
        assert config.confidence_threshold >= 0.5

        # PID parameters should be set
        assert config.pid_x is not None
        assert config.pid_y is not None
        assert config.pid_z is not None

        # Drone config should have safety settings
        assert drone_config.min_battery >= 10
        assert drone_config.max_speed <= 100

    def test_builder_creates_valid_config(self):
        """Test that builder always creates valid config."""
        config = (
            ConfigBuilder()
            .with_model("yolov8x")
            .with_confidence(0.8)
            .with_target_classes(["person", "car", "bicycle"])
            .with_drone_speed(40)
            .build()
        )

        # Should be valid Config instance
        assert isinstance(config, Config)

        # All fields should be set
        assert config.model_name is not None
        assert config.confidence_threshold is not None
        assert config.target_classes is not None
        assert config.drone_speed is not None

    def test_config_serialization(self):
        """Test that config can be represented as dict."""
        config = Config()

        # Should be able to access all attributes
        attrs = vars(config)

        assert "model_name" in attrs
        assert "confidence_threshold" in attrs
        assert "device" in attrs
        assert "drone_speed" in attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
