#!/usr/bin/env python3
"""
Drone Demo - Autonomous object tracking with DJI Tello.

This demo provides full autonomous tracking capabilities with the Tello drone.
The drone will automatically follow detected objects using PID control.

SAFETY WARNINGS:
- Always fly in an open area away from people and obstacles
- Keep manual control ready (spacebar to disable tracking)
- Monitor battery level
- Be prepared to emergency land (ESC key)

Controls:
    TAB       - Takeoff
    BACKSPACE - Land
    ESC       - Emergency stop
    SPACE     - Toggle tracking mode
    q         - Quit (will land first)

    Manual control (when tracking disabled):
    w/s       - Forward/Backward
    a/d       - Left/Right
    UP/DOWN   - Ascend/Descend
    LEFT/RIGHT- Rotate Left/Right

    Display:
    h         - Toggle HUD
    f         - Toggle FPS
    t         - Toggle telemetry
    r         - Record video
    c         - Take photo

Usage:
    python demo_drone.py
    python demo_drone.py --model yolov8s --confidence 0.6
    python demo_drone.py --classes person
    python demo_drone.py --mock  # Test without drone hardware
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, DroneConfig, get_drone_config
from src.detector import ObjectDetector
from src.drone_controller import DroneController, DroneState, MockDroneController
from src.tracker import SingleObjectTracker
from src.utils import (
    FPSCounter,
    draw_bbox,
    draw_crosshair,
    draw_info_panel,
    draw_trajectory,
    draw_vector,
)


class DroneDemo:
    """Drone tracking demo application."""

    def __init__(
        self, config: Config, drone_config: DroneConfig, use_mock: bool = False
    ):
        self.config = config
        self.drone_config = drone_config

        # Initialize components
        print("Initializing detector...")
        self.detector = ObjectDetector(config)
        print(f"Loaded {config.model_name} on {config.device}")

        self.tracker = SingleObjectTracker(config)
        self.fps_counter = FPSCounter()

        # Initialize drone controller
        if use_mock:
            print("Using mock drone controller (no hardware)")
            self.drone = MockDroneController(config, drone_config)
            self.use_mock = True
        else:
            print("Using real drone controller")
            self.drone = DroneController(config, drone_config)
            self.use_mock = False

        # State
        self.running = False
        self.show_hud = True
        self.show_fps = True
        self.show_telemetry = True
        self.recording = False
        self.video_writer = None

        # Manual control state
        self.manual_speed = 30

        self._print_controls()

    def _print_controls(self) -> None:
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("CONTROLS")
        print("=" * 60)
        print("Flight:")
        print("  TAB       - Takeoff")
        print("  BACKSPACE - Land")
        print("  ESC       - Emergency stop")
        print("  SPACE     - Toggle tracking mode")
        print("\nManual control (tracking off):")
        print("  w/s       - Forward/Backward")
        print("  a/d       - Left/Right")
        print("  UP/DOWN   - Ascend/Descend")
        print("  LEFT/RIGHT- Rotate Left/Right")
        print("\nDisplay:")
        print("  h - Toggle HUD")
        print("  f - Toggle FPS")
        print("  t - Toggle telemetry")
        print("  r - Record video")
        print("  c - Take photo")
        print("  q - Quit (will land first)")
        print("=" * 60)
        print()

    def start(self) -> None:
        """Start the demo."""
        # Connect to drone
        if not self.drone.connect():
            print("Failed to connect to drone")
            return

        print("\nDrone connected! Ready to fly.")
        print("Press TAB to takeoff when ready.")

        self.running = True
        self.run_loop()

    def run_loop(self) -> None:
        """Main processing loop."""
        while self.running:
            # Get frame
            if self.use_mock:
                # For mock, use webcam
                if not hasattr(self, "mock_cap"):
                    self.mock_cap = cv2.VideoCapture(0)
                ret, frame = self.mock_cap.read()
                if not ret:
                    continue
            else:
                frame = self.drone.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

            # Update FPS
            self.fps_counter.update()

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display
            cv2.imshow("Drone Tracking Demo", processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.handle_keypress(key)

        self.cleanup()

    def process_frame(self, frame):
        """Process a single frame."""
        display_frame = frame.copy()

        # Detect and track
        if self.drone.is_flying():
            detection = self.detector.detect_closest_to_center(frame)
            target = self.tracker.update(detection)

            # Autonomous tracking
            if self.drone.tracking_enabled:
                self.drone.track_target(target)

            # Visualize
            if target and target.disappeared == 0:
                color = (0, 255, 0) if self.drone.tracking_enabled else (255, 165, 0)
                label = f"{target.class_name} (ID: {target.id})"

                display_frame = draw_bbox(
                    display_frame,
                    target.bbox,
                    label,
                    target.confidence,
                    color,
                    thickness=2,
                )

                # Draw trajectory
                if len(target.centers) > 1:
                    display_frame = draw_trajectory(
                        display_frame, list(target.centers), (0, 255, 255), thickness=2
                    )

                # Draw tracking vector (to frame center)
                if self.drone.tracking_enabled:
                    h, w = frame.shape[:2]
                    frame_center = (w // 2, h // 2)
                    display_frame = draw_vector(
                        display_frame,
                        target.center,
                        frame_center,
                        (255, 0, 255),
                        thickness=2,
                    )

        # Draw frame center crosshair
        display_frame = draw_crosshair(display_frame, color=(0, 0, 255), size=30)

        # Draw HUD
        if self.show_hud:
            display_frame = self.draw_hud(display_frame)

        # Record if enabled
        if self.recording and self.video_writer:
            self.video_writer.write(display_frame)

        return display_frame

    def draw_hud(self, frame):
        """Draw heads-up display."""
        info = {}

        # FPS
        if self.show_fps:
            info["FPS"] = f"{self.fps_counter.get_fps():.1f}"

        # Drone state
        info["State"] = self.drone.state.value.upper()

        # Tracking state
        if self.drone.is_flying():
            tracking_status = "ACTIVE" if self.drone.tracking_enabled else "MANUAL"
            info["Mode"] = tracking_status

        # Target info
        if self.tracker.has_target():
            target = self.tracker.target
            info["Target"] = f"ID:{target.id}"
            info["Confidence"] = f"{target.confidence:.2f}"
        else:
            info["Target"] = "None"

        # Telemetry
        if self.show_telemetry and not self.use_mock:
            telemetry = self.drone.get_telemetry()
            info["Battery"] = f"{telemetry.get('battery', 0)}%"
            info["Height"] = f"{telemetry.get('height', 0)}cm"
            info["Temp"] = f"{telemetry.get('temperature', 0)}°C"

        # Recording indicator
        if self.recording:
            info["REC"] = "●"

        # Draw panel
        frame = draw_info_panel(
            frame,
            info,
            position="top-left",
            bg_color=(0, 0, 0),
            text_color=(0, 255, 0) if self.drone.tracking_enabled else (255, 165, 0),
            alpha=0.7,
        )

        return frame

    def handle_keypress(self, key: int) -> None:
        """Handle keyboard input."""
        # Flight controls
        if key == 9:  # TAB
            if self.drone.state == DroneState.CONNECTED:
                print("Taking off...")
                self.drone.takeoff()

        elif key == 8:  # BACKSPACE
            if self.drone.is_flying():
                print("Landing...")
                self.drone.land()

        elif key == 27:  # ESC
            print("EMERGENCY STOP!")
            self.drone.emergency_stop()
            self.running = False

        elif key == ord(" "):  # SPACE
            if self.drone.is_flying():
                if self.drone.tracking_enabled:
                    self.drone.disable_tracking()
                    print("Tracking disabled - manual control active")
                else:
                    self.drone.enable_tracking()
                    print("Tracking enabled - autonomous mode")

        # Manual controls (only when not tracking)
        elif key == ord("w") and not self.drone.tracking_enabled:
            self.drone.manual_control(fb=self.manual_speed)

        elif key == ord("s") and not self.drone.tracking_enabled:
            self.drone.manual_control(fb=-self.manual_speed)

        elif key == ord("a") and not self.drone.tracking_enabled:
            self.drone.manual_control(lr=-self.manual_speed)

        elif key == ord("d") and not self.drone.tracking_enabled:
            self.drone.manual_control(lr=self.manual_speed)

        elif key == 82 and not self.drone.tracking_enabled:  # UP arrow
            self.drone.manual_control(ud=self.manual_speed)

        elif key == 84 and not self.drone.tracking_enabled:  # DOWN arrow
            self.drone.manual_control(ud=-self.manual_speed)

        elif key == 81 and not self.drone.tracking_enabled:  # LEFT arrow
            self.drone.manual_control(yaw=-self.manual_speed)

        elif key == 83 and not self.drone.tracking_enabled:  # RIGHT arrow
            self.drone.manual_control(yaw=self.manual_speed)

        # Display controls
        elif key == ord("h"):
            self.show_hud = not self.show_hud
            print(f"HUD {'shown' if self.show_hud else 'hidden'}")

        elif key == ord("f"):
            self.show_fps = not self.show_fps
            print(f"FPS display {'shown' if self.show_fps else 'hidden'}")

        elif key == ord("t"):
            self.show_telemetry = not self.show_telemetry
            print(f"Telemetry {'shown' if self.show_telemetry else 'hidden'}")

        elif key == ord("r"):
            self.toggle_recording()

        elif key == ord("c"):
            self.take_photo()

        elif key == ord("q"):
            print("Quitting...")
            if self.drone.is_flying():
                print("Landing before quit...")
                self.drone.land()
            self.running = False

    def toggle_recording(self) -> None:
        """Toggle video recording."""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                self.config.fps,
                (self.config.frame_width, self.config.frame_height),
            )

            self.recording = True
            print(f"Recording started: {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.recording = False
            print("Recording stopped")

    def take_photo(self) -> None:
        """Take a photo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"

        frame = self.drone.get_frame()
        if frame is not None:
            cv2.imwrite(filename, frame)
            print(f"Photo saved: {filename}")
        else:
            print("Failed to capture photo")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.recording and self.video_writer:
            self.video_writer.release()

        if hasattr(self, "mock_cap"):
            self.mock_cap.release()

        self.drone.disconnect()
        cv2.destroyAllWindows()
        print("Demo stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Drone demo for autonomous object tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo_drone.py
                python demo_drone.py --model yolov8s --confidence 0.6
                python demo_drone.py --classes person
                python demo_drone.py --mock  # Test without drone
                Safety:
                - Always fly in open areas away from people
                - Monitor battery level
                - Keep emergency stop ready (ESC key)
                        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLO model to use (default: yolov8s)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Detection confidence threshold (default: 0.6)",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Target classes to detect (e.g., person ball)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run model on (default: auto)",
    )

    parser.add_argument(
        "--speed", type=int, default=50, help="Drone movement speed 0-100 (default: 50)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock drone (for testing without hardware)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configurations
    config, drone_config = get_drone_config()
    config.model_name = args.model
    config.confidence_threshold = args.confidence
    config.drone_speed = args.speed

    if args.classes:
        config.target_classes = args.classes

    if args.device:
        config.device = args.device

    # Safety check
    if not args.mock:
        print("\n" + "!" * 60)
        print("SAFETY WARNING")
        print("!" * 60)
        print("You are about to fly a real drone.")
        print("- Ensure you are in an open area")
        print("- Keep away from people and obstacles")
        print("- Monitor battery level")
        print("- Be ready to emergency stop (ESC key)")
        print("!" * 60)

        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted")
            return

    # Create and start demo
    demo = DroneDemo(config, drone_config, use_mock=args.mock)

    try:
        demo.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        demo.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        demo.cleanup()


if __name__ == "__main__":
    main()
