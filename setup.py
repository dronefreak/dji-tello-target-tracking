#!/usr/bin/env python3
"""Setup script for DJI Tello Target Tracking package.

Install in development mode:
    pip install -e .

Install with extras:
    pip install -e ".[dev]"
    pip install -e ".[all]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Real-time object detection and tracking for DJI Tello drones using PyTorch"


# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_file = Path(__file__).parent / filename
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#") and not line.startswith("-r")
            ]
    return []


# Core requirements
install_requires = read_requirements("requirements.txt")

# Development requirements
dev_requires = read_requirements("requirements-dev.txt")

# Optional extras
extras_require = {
    "dev": dev_requires,
    "test": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
    ],
    "docs": [
        "Sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "sphinx-autodoc-typehints>=1.24.0",
    ],
    "all": dev_requires,
}

setup(
    name="dji-tello-target-tracking",
    version="2.0.0",
    author="dronefreak",
    author_email="",
    description="Real-time object detection and tracking for DJI Tello drones using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dronefreak/dji-tello-target-tracking",
    project_urls={
        "Bug Reports": "https://github.com/dronefreak/dji-tello-target-tracking/issues",
        "Source": "https://github.com/dronefreak/dji-tello-target-tracking",
        "Documentation": "https://github.com/dronefreak/dji-tello-target-tracking/blob/main/README.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "drone",
        "tello",
        "dji",
        "object-detection",
        "object-tracking",
        "computer-vision",
        "pytorch",
        "yolo",
        "yolov8",
        "autonomous",
        "robotics",
    ],
    entry_points={
        "console_scripts": [
            "tello-track=demo_drone:main",
            "tello-demo=demo_webcam:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
