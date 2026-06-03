# AI Coding Agent Instructions for LeRobot

Welcome to the LeRobot codebase! This document provides essential guidance for AI coding agents to be productive in this repository. Follow these instructions to understand the architecture, workflows, and conventions specific to this project.

## Project Overview
LeRobot is a robotics library designed for advanced manipulation tasks. It includes tools for:
- **Robot control**: Teleoperation, calibration, and motor setup.
- **Policy training and evaluation**: Pretrained policies and custom training workflows.
- **Dataset management**: Video and image datasets for benchmarking and training.
- **Simulation and real-world integration**: Support for various robot models and environments.

### Key Components
- **`src/lerobot`**: Core library containing modules for:
  - `cameras/`: Camera utilities and integration.
  - `datasets/`: Dataset handling and preprocessing.
  - `policies/`: Policy models and training scripts.
  - `robots/`: Robot-specific configurations and utilities.
  - `utils/`: Shared utilities.
- **`benchmarks/video`**: Scripts and documentation for video encoding/decoding benchmarks.
- **`docs/`**: Documentation source files.
- **`examples/`**: Example scripts for common workflows.
- **`tests/`**: Unit and integration tests.

## Developer Workflows
### Building the Documentation
1. Install dependencies:
   ```bash
   pip install -e . -r docs-requirements.txt
   ```
2. Build the documentation:
   ```bash
   doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build
   ```

### Running Benchmarks
- Navigate to `benchmarks/video` and run:
  ```bash
  python run_video_benchmark.py
  ```
  Refer to `benchmarks/video/README.md` for detailed usage.

### Testing
- Run all tests:
  ```bash
  pytest tests/
  ```
- Test specific modules (e.g., cameras):
  ```bash
  pytest tests/cameras/
  ```

## Project-Specific Conventions
- **Documentation**: Use MDX format for `docs/source/`. Follow the structure in `_toctree.yml`.
- **Video Processing**: Refer to `benchmarks/video/README.md` for encoding/decoding standards.
- **Policy Training**: Policies are organized under `src/lerobot/policies/`. Each policy has its own README for usage.
- **Testing Artifacts**: Place test-specific data under `tests/artifacts/`.

## Integration Points
- **External Dependencies**:
  - `torchvision`, `ffmpegio`, `decord`: For video processing.
  - `pytest`: For testing.
- **Cross-Component Communication**:
  - Policies interact with datasets and robot configurations.
  - Camera utilities feed into datasets and benchmarks.

## Examples
- Load a dataset:
  ```python
  from lerobot.datasets import load_dataset
  dataset = load_dataset("lerobot/pusht_image")
  ```
- Train a policy:
  ```bash
  python examples/3_train_policy.py
  ```

## References
- [Main README](../README.md)
- [Video Benchmark README](../benchmarks/video/README.md)
- [Documentation Guide](../docs/README.md)

---
For further questions, refer to the respective module README files or the `docs/source/` directory.