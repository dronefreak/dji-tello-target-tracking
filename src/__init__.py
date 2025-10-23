"""DJI Tello Target Tracking Package.

A modern PyTorch-based object detection and tracking system for DJI Tello drones.
Supports both real-time drone tracking and standalone webcam demos.
"""

__version__ = "2.0.0"
__author__ = "dronefreak"
__license__ = "Apache-2.0"

from src.config import Config
from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.drone_controller import DroneController

__all__ = [
    "Config",
    "ObjectDetector",
    "ObjectTracker",
    "DroneController",
]
