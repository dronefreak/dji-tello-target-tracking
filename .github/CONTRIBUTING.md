# Contributing to DJI Tello Target Tracking

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to this project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/dronefreak/dji-tello-target-tracking/issues) to avoid duplicates.

**When submitting a bug report, include:**

- A clear and descriptive title
- Exact steps to reproduce the problem
- Expected behavior vs. actual behavior
- Screenshots or videos if applicable
- Your environment:
  - OS (Windows/Linux/macOS)
  - Python version
  - Installed package versions (`pip list`)
  - Drone model (if applicable)
  - GPU/CPU specs

**Template:**

```markdown
**Description:**
Brief description of the bug

**Steps to Reproduce:**

1. Step one
2. Step two
3. ...

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Environment:**

- OS: Ubuntu 22.04
- Python: 3.10.8
- PyTorch: 2.0.1
- CUDA: 11.8
- Drone: DJI Tello

**Additional Context:**
Any other relevant information
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any potential drawbacks or challenges
- Include mockups or examples if applicable

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

### Pull Requests

We actively welcome your pull requests! Here's the process:

1. Fork the repo and create your branch from `main`
2. Make your changes
3. Add tests for any new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/dji-tello-target-tracking.git
cd dji-tello-target-tracking

# Add upstream remote
git remote add upstream https://github.com/dronefreak/dji-tello-target-tracking.git
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/bug-description
```

### 4. Make Your Changes

Write your code, following our [coding standards](#coding-standards).

### 5. Test Your Changes

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_your_module.py -v
```

### 6. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add support for custom detection models"

# Or for bug fixes
git commit -m "fix: resolve tracking loss on fast movements"
```

**Commit Message Convention:**

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### 7. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatter**: Black
- **Import sorting**: isort
- **Type hints**: Encouraged for public APIs

### Code Formatting

```bash
# Format all code
make format

# This runs:
# - black (auto-formatter)
# - isort (import sorter)
```

### Linting

```bash
# Run all linters
make lint

# This runs:
# - flake8 (style guide enforcement)
# - pylint (advanced linting)
```

### Type Checking

```bash
# Run type checker
make type-check

# This runs mypy
```

### Code Quality Checklist

Before submitting, ensure:

- ✅ Code is formatted with Black
- ✅ Imports are sorted with isort
- ✅ No flake8 violations
- ✅ No pylint errors (warnings are okay)
- ✅ Type hints for public functions
- ✅ Docstrings for classes and public methods
- ✅ Tests pass
- ✅ New functionality has tests
- ✅ Documentation is updated

### Example Good Code

```python
"""Module for object detection."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class Detector:
    """Detect objects in images using YOLOv8.

    Args:
        model_name: Name of YOLO model to use
        confidence: Confidence threshold for detections

    Example:
        >>> detector = Detector("yolov8n", 0.5)
        >>> detections = detector.detect(frame)
    """

    def __init__(self, model_name: str = "yolov8n", confidence: float = 0.5):
        self.model_name = model_name
        self.confidence = confidence
        self._load_model()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of Detection objects

        Raises:
            ValueError: If frame is invalid
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")

        # Detection logic here
        return []

    def _load_model(self) -> None:
        """Load YOLO model (private method)."""
        # Implementation
        pass
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names

```python
def test_detector_returns_empty_list_for_blank_frame():
    """Test that detector returns empty list when given blank frame."""
    detector = ObjectDetector(Config())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert isinstance(detections, list)
    assert len(detections) == 0
```

### Test Coverage

- Aim for >80% code coverage
- Test both success and failure cases
- Test edge cases and boundary conditions
- Mock external dependencies (drone hardware, network calls)

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_tracker.py::TestObjectTracker::test_register_single_detection

# Run tests matching pattern
pytest -k "tracker"

# Run in parallel (faster)
pytest -n auto
```

## Documentation

### Code Documentation

- **Modules**: Docstring at the top explaining purpose
- **Classes**: Docstring with description, args, attributes, examples
- **Public methods**: Docstring with args, returns, raises
- **Private methods**: Optional docstring if complex

### Documentation Format

We use Google-style docstrings:

```python
def track_target(self, target: Optional[TrackedObject]) -> None:
    """Track a target object using PID control.

    This method computes control commands based on the target's position
    relative to the frame center and sends them to the drone.

    Args:
        target: TrackedObject to follow, or None if lost

    Raises:
        RuntimeError: If drone is not in tracking mode

    Example:
        >>> drone.enable_tracking()
        >>> drone.track_target(target)
    """
    # Implementation
```

### Updating Documentation

When adding features, update:

- **README.md**: Usage examples, features list
- **Code docstrings**: For new classes/functions
- **CHANGELOG.md**: Add entry for your changes
- **docs/**: Update detailed docs if applicable

## Pull Request Process

### Before Submitting

1. **Sync with upstream**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run quality checks**

   ```bash
   make quality
   ```

3. **Test thoroughly**

   ```bash
   make test-cov
   ```

4. **Update documentation**

### PR Guidelines

**Title Format:**

```
feat: add gesture recognition for drone control
fix: resolve tracking loss during rapid movements
docs: improve installation instructions
```

**Description Template:**

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to break)
- [ ] Documentation update

## Testing

Describe the tests you ran and their results

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, maintainers will merge

**Response Time:**

- Initial review: Within 1 week
- Follow-up reviews: Within 3 days

## Community

### Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/dronefreak/dji-tello-target-tracking/discussions)
- **Bugs**: Open an [Issue](https://github.com/dronefreak/dji-tello-target-tracking/issues)
- **Chat**: Join our community (if applicable)

### Recognition

Contributors will be:

- Listed in release notes
- Credited in documentation
- Added to CONTRIBUTORS.md (if significant contribution)

## Areas for Contribution

We especially welcome contributions in:

- 🎯 **New Features**: Multi-drone coordination, SLAM, gesture control
- 🐛 **Bug Fixes**: Any issues you encounter
- 📝 **Documentation**: Tutorials, examples, translations
- 🧪 **Testing**: More test coverage, hardware testing
- 🎨 **UX/UI**: Better visualization, mobile app
- ⚡ **Performance**: Optimization, speed improvements
- 🔧 **Hardware Support**: Other drone models

## Questions?

Don't hesitate to ask! Open a discussion or issue, and we'll help you get started.

Thank you for contributing! 🚀
