# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup

- **Installation**: `uv pip install -e .` (editable install)
- **With development dependencies**: `uv pip install -e ".[dev,test]"`
- **With all features**: `uv pip install -e ".[all]"`
- **Recommended**: `uv sync --extra dev --extra test` for dependency management

### Code Quality

- **Linting**: `uv run ruff check .` (configured in pyproject.toml)
- **Formatting**: `uv run ruff format .`
- **Pre-commit hooks**: `uv run pre-commit install` then `uv run pre-commit run --all-files`

### Testing

- **Run all tests**: `uv run python -m pytest -sv ./tests`
- **Run specific test file**: `uv run pytest tests/<TEST_TO_RUN>.py`
- **End-to-end tests**: `make test-end-to-end` (requires DEVICE=cpu or DEVICE=cuda)
- **Single policy tests**: `make test-act-ete-train`, `make test-diffusion-ete-eval`, etc.

### Training and Evaluation

- **Train policy**: `uv run python -m lerobot.scripts.train --policy.type=act --dataset.repo_id=lerobot/pusht`
- **Evaluate policy**: `uv run python -m lerobot.scripts.eval --policy.path=path/to/model`
- **Resume training**: `uv run python -m lerobot.scripts.train --resume=true --config_path=path/to/train_config.json`
- **With WandB logging**: Add `--wandb.enable=true` to training commands

### Data and Visualization

- **Visualize dataset**: `uv run python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0`
- **Display system info**: `uv run python -m lerobot.scripts.display_sys_info`

### Robot Hardware Commands

- **Calibrate robot**: `uv run lerobot-calibrate`
- **Setup motors**: `uv run lerobot-setup-motors`
- **Record demonstrations**: `uv run lerobot-record`
- **Replay demonstrations**: `uv run lerobot-replay`
- **Teleoperate robot**: `uv run lerobot-teleoperate`
- **Find cameras**: `uv run lerobot-find-cameras`
- **Find serial ports**: `uv run lerobot-find-port`

## Code Architecture

### Core Structure

- **src/lerobot/**: Main package containing all source code
- **policies/**: Neural network policies (ACT, Diffusion, TDMPC, VQ-BeT, SmolVLA, Pi0)
- **datasets/**: Dataset loading, transforms, and LeRobotDataset format
- **robots/**: Hardware integration for various robot platforms
- **envs/**: Simulation environment wrappers
- **scripts/**: Training, evaluation, and utility scripts
- **transport/**: gRPC-based client-server architecture for async inference

### Policy Architecture

Policies follow a standard interface:

- `forward()`: Main forward pass for training/inference
- `predict()`: Generate actions from observations
- Configuration classes inherit from policy-specific configs
- Models saved in HuggingFace format (safetensors + config.json)

### Dataset System

- **LeRobotDataset**: Core dataset class supporting temporal indexing with `delta_timestamps`
- **Data formats**: Video (MP4) + parquet metadata for efficiency
- **Statistics**: Automatic computation of normalization stats per feature
- **Episodes**: Data organized by episodes with start/end indices

### Robot Integration

- Modular robot support through base `Robot` class
- **Supported robots**: SO-100/101, Koch, ALOHA, HopeJR, LeKiwi, Stretch3, ViperX
- Motor controllers: Dynamixel, Feetech servo support
- Camera systems: OpenCV, Intel RealSense integration

### Async Architecture (Client-Server)

- **Policy Server**: gRPC server for remote policy inference (policy_server.py)
- **Robot Client**: Connects robots to remote policy server (robot_client.py)
- Supports real-time inference with configurable FPS and latency
- Queue-based observation/action handling with timeout management

### Configuration System

Uses `draccus` for hierarchical configuration management:

- **Train configs**: Complete training pipeline configuration
- **Policy configs**: Model-specific hyperparameters
- **Environment configs**: Simulation environment settings
- YAML/JSON config file support with command-line overrides

## Important Implementation Notes

### Policy Development

- All policies must have a `name` class attribute
- Update `available_policies` in `src/lerobot/__init__.py` when adding new policies
- Follow the configuration pattern: separate `configuration_*.py` and `modeling_*.py` files
- Use the PreTrainedPolicy base class for HuggingFace Hub integration

### Dataset Development

- Update `available_datasets_per_env` in `src/lerobot/__init__.py`
- Follow LeRobotDataset format for consistency
- Support both local and HuggingFace Hub storage
- Include proper episode segmentation and temporal indexing

### Environment Integration

- Update `available_tasks_per_env` and `available_datasets_per_env` in `src/lerobot/__init__.py`
- Environments are typically external packages (gym-aloha, gym-pusht, gym-xarm)
- Follow Gymnasium API standards

### Hardware Integration

- Robot configs located in respective robot subdirectories
- Use factory pattern for robot/camera/motor instantiation
- Calibration data stored in robot-specific format
- Support both teleoperation and autonomous control modes

### Testing Strategy

- Unit tests for individual components in `tests/`
- End-to-end tests via Makefile for complete training/eval pipelines
- Hardware mocking for CI/CD compatibility
- Artifact-based testing with git-lfs for model comparisons

This codebase emphasizes modularity, reproducibility, and ease of use for both simulation and real-world robotics applications.
