"""Drone controller for DJI Tello with autonomous tracking.

Handles connection, movement, and PID-based target following.
"""

import cv2
import numpy as np
import time
from typing import Optional
from enum import Enum

try:
    from djitellopy import Tello

    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False
    print("Warning: djitellopy not installed. Install with: pip install djitellopy")

from src.config import Config, DroneConfig
from src.tracker import TrackedObject
from src.utils import PIDController


class DroneState(Enum):
    """Drone state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    HOVERING = "hovering"
    TRACKING = "tracking"
    LANDING = "landing"
    EMERGENCY = "emergency"


class DroneController:
    """High-level controller for DJI Tello drone.

    Provides tracking automation with PID control.
    """

    def __init__(self, config: Config, drone_config: DroneConfig):
        if not TELLO_AVAILABLE:
            raise ImportError("djitellopy not installed. " "Install with: pip install djitellopy")

        self.config = config
        self.drone_config = drone_config
        self.drone: Optional[Tello] = None
        self.state = DroneState.DISCONNECTED

        # PID controllers for each axis
        self.pid_x = PIDController(*config.pid_x, output_limits=(-100, 100))
        self.pid_y = PIDController(*config.pid_y, output_limits=(-100, 100))
        self.pid_z = PIDController(*config.pid_z, output_limits=(-100, 100))
        self.pid_yaw = PIDController(*config.pid_yaw, output_limits=(-100, 100))

        # Tracking state
        self.tracking_enabled = False
        self.frame_center = (config.frame_width // 2, config.frame_height // 2)

        # Safety
        self.last_command_time = time.time()
        self.command_interval = 0.1  # Min time between commands

    def connect(self) -> bool:
        """Connect to Tello drone.

        Returns:
            True if connection successful
        """
        try:
            print("Connecting to Tello...")
            self.drone = Tello()
            self.drone.connect()

            # Get drone info
            battery = self.drone.get_battery()
            print(f"Connected! Battery: {battery}%")

            if battery < self.drone_config.min_battery:
                print(
                    f"WARNING: Battery too low ({battery}%). Minimum: {self.drone_config.min_battery}%"
                )
                return False

            self.state = DroneState.CONNECTED

            # Start video stream
            self.drone.streamon()
            time.sleep(2)  # Wait for stream to stabilize

            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            self.state = DroneState.DISCONNECTED
            return False

    def disconnect(self) -> None:
        """Disconnect from drone."""
        if self.drone is not None:
            try:
                if self.state not in [DroneState.DISCONNECTED, DroneState.CONNECTED]:
                    self.land()

                self.drone.streamoff()
                self.drone.end()
                print("Disconnected from drone")
            except:
                pass

            self.state = DroneState.DISCONNECTED
            self.drone = None

    def takeoff(self) -> bool:
        """Takeoff sequence.

        Returns:
            True if successful
        """
        if self.state != DroneState.CONNECTED:
            print("Drone not connected")
            return False

        try:
            print("Taking off...")
            self.drone.takeoff()
            time.sleep(3)  # Wait for stabilization
            self.state = DroneState.HOVERING
            print("Takeoff successful")
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        """Landing sequence.

        Returns:
            True if successful
        """
        if self.state == DroneState.DISCONNECTED:
            return False

        try:
            print("Landing...")
            self.state = DroneState.LANDING
            self.tracking_enabled = False

            self.drone.send_rc_control(0, 0, 0, 0)  # Stop all movement
            time.sleep(0.5)

            self.drone.land()
            time.sleep(3)

            self.state = DroneState.CONNECTED
            print("Landed successfully")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            return False

    def emergency_stop(self) -> None:
        """Emergency stop - cuts motors immediately."""
        if self.drone is not None:
            try:
                print("EMERGENCY STOP!")
                self.drone.emergency()
                self.state = DroneState.EMERGENCY
            except:
                pass

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from drone camera.

        Returns:
            Frame as numpy array (BGR) or None
        """
        if self.drone is None or self.state == DroneState.DISCONNECTED:
            return None

        try:
            frame = self.drone.get_frame_read().frame
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except:
            return None

    def enable_tracking(self) -> None:
        """Enable autonomous tracking mode."""
        if self.state == DroneState.HOVERING:
            self.tracking_enabled = True
            self.state = DroneState.TRACKING

            # Reset PIDs
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
            self.pid_yaw.reset()

            print("Tracking enabled")

    def disable_tracking(self) -> None:
        """Disable autonomous tracking."""
        self.tracking_enabled = False
        if self.state == DroneState.TRACKING:
            self.state = DroneState.HOVERING
            self.send_rc_control(0, 0, 0, 0)  # Stop movement
            print("Tracking disabled")

    def track_target(self, target: Optional[TrackedObject]) -> None:
        """Track a target object using PID control.

        Args:
            target: TrackedObject to follow, or None if lost
        """
        if not self.tracking_enabled or self.state != DroneState.TRACKING:
            return

        # If target lost, hover in place
        if target is None or target.disappeared > 0:
            self.send_rc_control(0, 0, 0, 0)
            return

        # Calculate errors from frame center
        error_x = target.center[0] - self.frame_center[0]
        error_y = target.center[1] - self.frame_center[1]

        # Calculate bbox size for distance estimation
        bbox_width = target.bbox[2] - target.bbox[0]
        bbox_height = target.bbox[3] - target.bbox[1]
        bbox_area = bbox_width * bbox_height

        # Normalize area (assume optimal tracking at ~20% of frame)
        frame_area = self.config.frame_width * self.config.frame_height
        target_area = frame_area * 0.2
        error_z = (bbox_area - target_area) / target_area * 100

        # Compute PID outputs
        cmd_lr = -self.pid_x.update(error_x)  # Left/Right (negative because x increases right)
        cmd_fb = self.pid_z.update(error_z)  # Forward/Backward
        cmd_ud = -self.pid_y.update(error_y)  # Up/Down (negative because y increases down)
        cmd_yaw = -self.pid_x.update(error_x) * 0.5  # Yaw follows x error

        # Apply dead zone
        threshold = self.config.movement_threshold
        if abs(error_x) < threshold:
            cmd_lr = 0
            cmd_yaw = 0
        if abs(error_y) < threshold:
            cmd_ud = 0
        if abs(error_z) < threshold:
            cmd_fb = 0

        # Send commands
        self.send_rc_control(int(cmd_lr), int(cmd_fb), int(cmd_ud), int(cmd_yaw))

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Send RC control command to drone.

        Args:
            lr: Left/Right velocity (-100 to 100)
            fb: Forward/Backward velocity (-100 to 100)
            ud: Up/Down velocity (-100 to 100)
            yaw: Yaw velocity (-100 to 100)
        """
        if self.drone is None:
            return

        # Rate limiting
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return

        # Clamp values
        lr = np.clip(lr, -100, 100)
        fb = np.clip(fb, -100, 100)
        ud = np.clip(ud, -100, 100)
        yaw = np.clip(yaw, -100, 100)

        try:
            self.drone.send_rc_control(lr, fb, ud, yaw)
            self.last_command_time = current_time
        except Exception as e:
            print(f"Failed to send command: {e}")

    def manual_control(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> None:
        """Manual control in hovering mode."""
        if self.state == DroneState.HOVERING:
            self.send_rc_control(lr, fb, ud, yaw)

    def get_telemetry(self) -> dict:
        """Get drone telemetry data.

        Returns:
            Dictionary with telemetry info
        """
        if self.drone is None or self.state == DroneState.DISCONNECTED:
            return {}

        try:
            return {
                "battery": self.drone.get_battery(),
                "temperature": self.drone.get_temperature(),
                "height": self.drone.get_height(),
                "barometer": self.drone.get_barometer(),
                "flight_time": self.drone.get_flight_time(),
                "speed_x": self.drone.get_speed_x(),
                "speed_y": self.drone.get_speed_y(),
                "speed_z": self.drone.get_speed_z(),
            }
        except:
            return {}

    def is_connected(self) -> bool:
        """Check if drone is connected."""
        return self.state != DroneState.DISCONNECTED

    def is_flying(self) -> bool:
        """Check if drone is in flight."""
        return self.state in [DroneState.HOVERING, DroneState.TRACKING]


class MockDroneController(DroneController):
    """Mock drone controller for testing without hardware.

    Simulates drone behavior.
    """

    def __init__(self, config: Config, drone_config: DroneConfig):
        super().__init__(config, drone_config)
        self.mock_battery = 100
        self.mock_height = 0
        self.mock_flying = False

    def connect(self) -> bool:
        print("Mock: Connected to simulated drone")
        self.state = DroneState.CONNECTED
        return True

    def disconnect(self) -> None:
        print("Mock: Disconnected")
        self.state = DroneState.DISCONNECTED

    def takeoff(self) -> bool:
        print("Mock: Taking off")
        self.state = DroneState.HOVERING
        self.mock_flying = True
        self.mock_height = 100
        return True

    def land(self) -> bool:
        print("Mock: Landing")
        self.state = DroneState.CONNECTED
        self.mock_flying = False
        self.mock_height = 0
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        # Return None - user should use webcam for testing
        return None

    def get_telemetry(self) -> dict:
        return {
            "battery": self.mock_battery,
            "height": self.mock_height,
            "temperature": 25,
            "flight_time": 0,
        }

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        # Simulate command
        pass
