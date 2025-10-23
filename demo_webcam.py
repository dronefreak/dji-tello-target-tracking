#!/usr/bin/env python3
"""
Webcam Demo - Test object detection and tracking without a drone.

This demo allows you to test the detection and tracking system using
just your webcam. Perfect for development and testing.

Controls:
    q - Quit
    t - Toggle tracking mode
    d - Toggle detector (YOLO/HSV)
    r - Reset tracker
    s - Save screenshot
    h - Toggle HUD display
    f - Toggle FPS display

Usage:
    python demo_webcam.py
    python demo_webcam.py --model yolov8s --confidence 0.6
    python demo_webcam.py --webcam 1
    python demo_webcam.py --video path/to/video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, get_webcam_config
from src.detector import HSVDetector, ObjectDetector
from src.tracker import SingleObjectTracker
from src.utils import (
    FPSCounter,
    draw_bbox,
    draw_crosshair,
    draw_info_panel,
    draw_trajectory,
    draw_vector,
)


class WebcamDemo:
    """Webcam demo application."""

    def __init__(self, config: Config, video_source: int = 0, video_file: str = None):
        self.config = config
        self.video_source = video_source
        self.video_file = video_file

        # Initialize components
        print("Initializing detector...")
        try:
            self.detector = ObjectDetector(config)
            self.use_yolo = True
            print(f"Loaded {config.model_name} on {config.device}")
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            print("Falling back to HSV detector")
            self.detector = HSVDetector(config, "green")
            self.use_yolo = False

        self.tracker = SingleObjectTracker(config)
        self.fps_counter = FPSCounter()

        # State
        self.tracking_enabled = True
        self.show_hud = True
        self.show_fps = True
        self.running = False

        # Video capture
        self.cap = None

        print("\nControls:")
        print("  q - Quit")
        print("  t - Toggle tracking")
        print("  d - Toggle detector (YOLO/HSV)")
        print("  r - Reset tracker")
        print("  s - Save screenshot")
        print("  h - Toggle HUD")
        print("  f - Toggle FPS")
        print()

    def start(self) -> None:
        """Start the demo."""
        # Open video source
        if self.video_file:
            print(f"Opening video file: {self.video_file}")
            self.cap = cv2.VideoCapture(self.video_file)
        else:
            print(f"Opening webcam {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            print("Error: Could not open video source")
            return

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        print("Demo started! Press 'q' to quit\n")
        self.running = True
        self.run_loop()

    def run_loop(self) -> None:
        """Main processing loop."""
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                if self.video_file:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Error: Failed to read frame")
                    break

            # Update FPS
            self.fps_counter.update()

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display
            cv2.imshow("Webcam Demo - Object Tracking", processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.handle_keypress(key)

        self.cleanup()

    def process_frame(self, frame):
        """Process a single frame."""
        display_frame = frame.copy()

        # Detect objects
        if self.tracking_enabled:
            if self.use_yolo:
                detection = self.detector.detect_closest_to_center(frame)
            else:
                detection = self.detector.detect(frame)

            # Update tracker
            target = self.tracker.update(detection)

            # Visualize
            if target and target.disappeared == 0:
                # Draw bounding box
                color = (0, 255, 0)
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

                # Draw velocity vector
                velocity = target.get_velocity()
                if velocity:
                    vx, vy = velocity
                    end_point = (
                        int(target.center[0] + vx * 20),
                        int(target.center[1] + vy * 20),
                    )
                    display_frame = draw_vector(
                        display_frame,
                        target.center,
                        end_point,
                        (255, 0, 255),
                        thickness=2,
                    )

        # Draw frame center crosshair
        display_frame = draw_crosshair(display_frame, color=(0, 0, 255), size=30)

        # Draw HUD
        if self.show_hud:
            display_frame = self.draw_hud(display_frame)

        return display_frame

    def draw_hud(self, frame):
        """Draw heads-up display."""
        info = {}

        if self.show_fps:
            info["FPS"] = f"{self.fps_counter.get_fps():.1f}"

        info["Tracking"] = "ON" if self.tracking_enabled else "OFF"
        info["Detector"] = "YOLO" if self.use_yolo else "HSV"

        if self.tracker.has_target():
            target = self.tracker.target
            info["Target"] = f"ID:{target.id} Age:{target.age}"
            info["Confidence"] = f"{target.confidence:.2f}"
        else:
            info["Target"] = "None"

        frame = draw_info_panel(
            frame,
            info,
            position="top-left",
            bg_color=(0, 0, 0),
            text_color=(0, 255, 0),
            alpha=0.6,
        )

        return frame

    def handle_keypress(self, key: int) -> None:
        """Handle keyboard input."""
        if key == ord("q"):
            print("Quitting...")
            self.running = False

        elif key == ord("t"):
            self.tracking_enabled = not self.tracking_enabled
            status = "enabled" if self.tracking_enabled else "disabled"
            print(f"Tracking {status}")
            if not self.tracking_enabled:
                self.tracker.reset()

        elif key == ord("d"):
            if self.use_yolo:
                print("Switching to HSV detector")
                self.detector = HSVDetector(self.config, "green")
                self.use_yolo = False
            else:
                try:
                    print("Switching to YOLO detector")
                    self.detector = ObjectDetector(self.config)
                    self.use_yolo = True
                except Exception as e:
                    print(f"Failed to switch to YOLO: {e}")

        elif key == ord("r"):
            print("Resetting tracker")
            self.tracker.reset()

        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

        elif key == ord("h"):
            self.show_hud = not self.show_hud
            print(f"HUD {'shown' if self.show_hud else 'hidden'}")

        elif key == ord("f"):
            self.show_fps = not self.show_fps
            print(f"FPS display {'shown' if self.show_fps else 'hidden'}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Demo stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Webcam demo for object detection and tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_webcam.py
  python demo_webcam.py --model yolov8s --confidence 0.6
  python demo_webcam.py --webcam 1
  python demo_webcam.py --video test_video.mp4
  python demo_webcam.py --classes person ball
        """,
    )

    parser.add_argument(
        "--webcam", type=int, default=0, help="Webcam device ID (default: 0)"
    )

    parser.add_argument(
        "--video", type=str, default=None, help="Path to video file (instead of webcam)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLO model to use (default: yolov8n)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
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
        "--width", type=int, default=960, help="Frame width (default: 960)"
    )

    parser.add_argument(
        "--height", type=int, default=720, help="Frame height (default: 720)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration
    config = get_webcam_config()
    config.model_name = args.model
    config.confidence_threshold = args.confidence
    config.frame_width = args.width
    config.frame_height = args.height

    if args.classes:
        config.target_classes = args.classes

    if args.device:
        config.device = args.device

    # Create and start demo
    demo = WebcamDemo(config, args.webcam, args.video)

    try:
        demo.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
