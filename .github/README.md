# DJI Tello Target Tracking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Real-time object detection and autonomous tracking for DJI Tello drones using PyTorch and YOLOv8. This modernized version replaces the old HSV color tracking with state-of-the-art deep learning models for robust, accurate object detection and tracking.

## ✨ Features

- 🎯 **Modern Object Detection**: YOLOv8-powered detection (person, ball, car, etc.)
- 🔄 **Robust Tracking**: Centroid-based multi-object tracking with trajectory prediction
- 🚁 **Autonomous Flight**: PID-controlled smooth target following
- 🎮 **Manual Override**: Instant switch between autonomous and manual control
- 📹 **Webcam Demo**: Test detection/tracking without drone hardware
- 🎨 **Rich Visualization**: Real-time HUD, bounding boxes, trajectories, velocity vectors
- 🛡️ **Safety First**: Battery monitoring, emergency stop, configurable limits
- 🧪 **Well Tested**: 165+ unit tests with >85% coverage

## 🎥 Demo

### Webcam Demo (No Hardware Required)

```bash
python demo_webcam.py
```

### Drone Tracking

```bash
python demo_drone.py
```

## 📋 Requirements

- Python 3.8+
- DJI Tello drone (for drone demos)
- CUDA-capable GPU (optional, for faster inference)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/dronefreak/dji-tello-target-tracking.git
cd dji-tello-target-tracking

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Test Without Drone (Webcam Demo)

```bash
# Run with default settings (YOLOv8n, detect all classes)
python demo_webcam.py

# Use a better model
python demo_webcam.py --model yolov8s --confidence 0.6

# Track specific objects only
python demo_webcam.py --classes person ball

# Use a video file instead of webcam
python demo_webcam.py --video test_video.mp4
```

**Webcam Demo Controls:**

- `q` - Quit
- `t` - Toggle tracking on/off
- `d` - Switch detector (YOLO/HSV)
- `r` - Reset tracker
- `s` - Save screenshot
- `h` - Toggle HUD
- `f` - Toggle FPS display

### 3. Fly the Drone

```bash
# First, test with mock drone (uses webcam, no hardware)
python demo_drone.py --mock

# When ready, fly for real
python demo_drone.py

# With custom settings
python demo_drone.py --model yolov8s --confidence 0.6 --speed 50
```

**⚠️ SAFETY WARNING**: Always fly in open areas away from people and obstacles!

**Drone Demo Controls:**

- `TAB` - Takeoff
- `BACKSPACE` - Land
- `ESC` - Emergency stop
- `SPACE` - Toggle autonomous tracking
- `w/s/a/d` - Manual control (forward/back/left/right)
- `↑/↓` - Manual altitude control
- `←/→` - Manual rotation
- `r` - Record video
- `c` - Take photo
- `q` - Quit (lands first)

## 📖 Usage Examples

### Basic Detection

```python
from src.config import Config
from src.detector import ObjectDetector
import cv2

# Initialize
config = Config()
config.model_name = "yolov8n"
config.confidence_threshold = 0.5
detector = ObjectDetector(config)

# Detect objects
frame = cv2.imread("test.jpg")
detections = detector.detect(frame)

for det in detections:
    print(f"Found {det.class_name} at {det.bbox} with confidence {det.confidence:.2f}")
```

### Single Object Tracking

```python
from src.config import Config
from src.detector import ObjectDetector
from src.tracker import SingleObjectTracker
import cv2

config = Config()
detector = ObjectDetector(config)
tracker = SingleObjectTracker(config)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect closest object to center
    detection = detector.detect_closest_to_center(frame)

    # Update tracker
    target = tracker.update(detection)

    if target and target.disappeared == 0:
        print(f"Tracking ID {target.id} at {target.center}")

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### Autonomous Drone Tracking

```python
from src.config import get_drone_config
from src.detector import ObjectDetector
from src.tracker import SingleObjectTracker
from src.drone_controller import DroneController

# Initialize
config, drone_config = get_drone_config()
detector = ObjectDetector(config)
tracker = SingleObjectTracker(config)
drone = DroneController(config, drone_config)

# Connect and takeoff
drone.connect()
drone.takeoff()

# Enable autonomous tracking
drone.enable_tracking()

while drone.is_flying():
    frame = drone.get_frame()
    detection = detector.detect_closest_to_center(frame)
    target = tracker.update(detection)

    # Drone automatically follows target
    drone.track_target(target)

drone.land()
drone.disconnect()
```

## 🏗️ Project Structure

```
dji-tello-target-tracking/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── detector.py         # YOLOv8 & HSV detection
│   ├── tracker.py          # Object tracking algorithms
│   ├── drone_controller.py # Tello drone interface
│   └── utils.py            # Helper functions
├── tests/
│   ├── test_config.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_utils.py
├── docs/                   # Documentation
├── demo_webcam.py          # Webcam demo script
├── demo_drone.py           # Drone tracking script
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## 🔧 Configuration

### Pre-configured Profiles

```python
from src.config import get_webcam_config, get_drone_config

# Optimized for webcam testing (fast, lower accuracy)
config = get_webcam_config()

# Optimized for drone tracking (accurate, outdoor use)
config, drone_config = get_drone_config()
```

### Custom Configuration

```python
from src.config import ConfigBuilder

config = (ConfigBuilder()
    .with_model("yolov8m")           # Use medium model
    .with_confidence(0.7)            # Higher confidence threshold
    .with_target_classes(["person"]) # Only track persons
    .with_drone_speed(60)            # Faster movement
    .build())
```

### Available YOLO Models

| Model   | Speed  | Accuracy   | Use Case               |
| ------- | ------ | ---------- | ---------------------- |
| yolov8n | ⚡⚡⚡ | ⭐⭐       | Webcam demos, testing  |
| yolov8s | ⚡⚡   | ⭐⭐⭐     | Default drone tracking |
| yolov8m | ⚡     | ⭐⭐⭐⭐   | High accuracy tracking |
| yolov8l | 🐌     | ⭐⭐⭐⭐⭐ | Maximum accuracy       |
| yolov8x | 🐌🐌   | ⭐⭐⭐⭐⭐ | Best possible accuracy |

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_tracker.py

# Run with verbose output
pytest -v

# Or use make commands
make test
make test-cov
```

## 🛠️ Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Format code
make format

# Run linters
make lint

# Run type checking
make type-check

# Run all quality checks
make quality
```

## 📊 Performance

### Detection Speed (640x480, CPU i7-10750H)

| Model   | FPS | mAP@50 |
| ------- | --- | ------ |
| YOLOv8n | 45  | 0.503  |
| YOLOv8s | 30  | 0.530  |
| YOLOv8m | 20  | 0.557  |

### Detection Speed (640x480, GPU RTX 3060)

| Model   | FPS | mAP@50 |
| ------- | --- | ------ |
| YOLOv8n | 120 | 0.503  |
| YOLOv8s | 95  | 0.530  |
| YOLOv8m | 70  | 0.557  |

## 🆚 Comparison with Original (v1.0)

| Feature       | v1.0 (2020)       | v2.0 (2025)                  |
| ------------- | ----------------- | ---------------------------- |
| Detection     | HSV color masking | YOLOv8 deep learning         |
| Objects       | Single color only | 80+ object classes           |
| Tracking      | Basic centroid    | Multi-object with prediction |
| Control       | Manual PID tuning | Auto-tuned PID               |
| Testing       | No tests          | 165+ unit tests              |
| Documentation | Basic README      | Full docs + examples         |
| Code Quality  | Mixed style       | Black formatted, typed       |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy) for the Tello drone interface
- Original project inspiration from PyImageSearch ball tracking tutorial

## 📮 Contact

- GitHub: [@dronefreak](https://github.com/dronefreak)
- Issues: [GitHub Issues](https://github.com/dronefreak/dji-tello-target-tracking/issues)

## ⚠️ Safety Disclaimer

**IMPORTANT**: This software controls a flying drone. Always:

- Fly in open, outdoor areas away from people and obstacles
- Follow local drone regulations and laws
- Monitor battery levels (lands automatically at 10%)
- Keep manual control ready at all times
- Practice in mock mode before real flights
- Never fly over people or near airports
- Be prepared to use emergency stop (ESC key)

The authors are not responsible for any damage or injury caused by the use of this software.

## 🗺️ Roadmap

- [ ] Add support for more drone models (DJI Mini, Mavic)
- [ ] Implement SLAM for indoor navigation
- [ ] Add gesture recognition for control
- [ ] Multi-drone coordinated tracking
- [ ] Real-time trajectory optimization
- [ ] Mobile app for remote monitoring
- [ ] Cloud training pipeline for custom models

## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{dji_tello_tracking_2025,
  author = {dronefreak},
  title = {DJI Tello Target Tracking with YOLOv8},
  year = {2025},
  url = {https://github.com/dronefreak/dji-tello-target-tracking},
  version = {2.0.0}
}
```

---

**Star ⭐ this repo if you find it useful!**
