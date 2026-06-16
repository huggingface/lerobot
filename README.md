# LeRobot

English | [中文](https://github.com/Hiwonder/LeRobot/blob/main/README_cn.md)

<p align="center">
  <img src="./sources/images/lerobot.png" alt="LeRobot Logo" width="600"/>
</p>

## Product Overview

LeRobot is a state-of-the-art AI robotics library developed by Hugging Face, adapted by Hiwonder for educational and research purposes. It provides models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry to robotics so that everyone can contribute and benefit from sharing datasets and pretrained models.

LeRobot contains state-of-the-art approaches that have been shown to transfer to the real-world with a focus on imitation learning and reinforcement learning. It supports various robot platforms including SO-101, HopeJR, LeKiwi, and more.

## Hiwonder SO-ARM101

Using an end-to-end imitation learning pipeline, you demonstrate tasks with a *Leader Arm*. The recorded trajectories are converted into a trained model, enabling the *Follower Arm* to reproduce the task autonomously.

<p align="center">
  <img src="./sources/images/模仿学习.gif" alt="Imitation Learning" width="600"/>
</p>

To support stable and repeatable experiments, Hiwonder SO-ARM101 features:

1. **Full access to the open-source hardware and software stack**: It can be trained using techniques like imitation learning and reinforcement learning to perform tasks such as object manipulation and motion replication. The entire system—from hardware to software and algorithms—is fully open-source.

<p align="center">
  <img src="./sources/images/so-arm101-opensource.png" alt="SO-ARM101 Open Source" width="600"/>
</p>

2. **30kg magnetic-encoder servos for high torque, low jitter, and precise control**: Combined with rotating base and flexible joint movements, the arm achieves smooth and fluid motion, eliminating the power limitations and vibrations found in the original design.

<p align="center">
  <img src="./sources/images/舵机.gif" alt="SO-ARM101 Servo" width="600"/>
</p>

3. **Dual-camera vision system for real-world perception and vision-based learning**: Precise close-range manipulation + global environment awareness.

<p align="center">
  <img src="./sources/images/双摄系统.gif" alt="SO-ARM101 Dual Camera" width="600"/>
</p>

SO-ARM101 provides a practical platform for studying learning-from-demonstration, manipulation, and embodied intelligence.

## Official Resources

### Official Hiwonder
- **Official Website**: [https://www.hiwonder.com/](https://www.hiwonder.com/)
- **Product Page**: [https://www.hiwonder.com/products/lerobot-so-101](https://www.hiwonder.com/products/lerobot-so-101)
- **Official Documentation**: [https://www.hiwonder.com.cn/store/learn/185.html](https://www.hiwonder.com.cn/store/learn/185.html)
- **Video Tutorial**: [https://www.youtube.com/watch?v=oitT8geMat0](https://www.youtube.com/watch?v=oitT8geMat0)
- **Technical Support**: support@hiwonder.com

### Original LeRobot
- **Original Repository**: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **Documentation**: [https://huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot)
- **Hugging Face Hub**: [https://huggingface.co/lerobot](https://huggingface.co/lerobot)

## Key Features

### AI Vision Functions
- **Object Detection** - Real-time object detection and recognition
- **Visual Grasping** - Hand-eye coordination for precise manipulation
- **Target Tracking** - Advanced object tracking with AI algorithms
- **AprilTag Recognition** - Precision tag-based positioning

### Imitation Learning
- **ACT Policy** - Action Chunking with Transformers
- **Diffusion Policy** - Diffusion-based action generation
- **VQ-BeT** - Vector Quantized Behavior Transformer
- **TDMPC** - Temporal Difference Model Predictive Control

### Robot Platforms
- **SO-101** - Affordable robotic arm (€114 per arm)
- **HopeJR** - Humanoid robot arm and hand for dexterous manipulation
- **LeKiwi** - Mobile robot platform with wheels
- **ALOHA** - Bimanual teleoperation system

### Programming Interface
- **Python SDK** - Complete Python programming interface
- **PyTorch Integration** - Deep learning framework support
- **Hugging Face Hub** - Dataset and model sharing platform
- **WandB Support** - Experiment tracking and visualization

## Hardware Configuration
- **Processor**: Compatible with various platforms (Raspberry Pi, PC, etc.)
- **Vision System**: USB cameras, depth cameras
- **Motor System**: Feetech servos, Dynamixel servos
- **Communication**: USB, Serial, WiFi

## Project Structure

```
lerobot/
├── src/lerobot/              # Core library
│   ├── cameras/              # Camera drivers and utilities
│   ├── configs/              # Configuration files
│   ├── datasets/             # Dataset handling
│   ├── envs/                 # Simulation environments
│   ├── model/                # Neural network models
│   ├── motors/               # Motor control drivers
│   ├── policies/             # Policy implementations
│   │   ├── act/              # ACT policy
│   │   ├── diffusion/        # Diffusion policy
│   │   ├── tdmpc/            # TDMPC policy
│   │   └── vqbet/            # VQ-BeT policy
│   ├── robots/               # Robot configurations
│   ├── scripts/              # Utility scripts
│   ├── teleoperators/        # Teleoperation systems
│   └── utils/                # Common utilities
├── examples/                 # Example scripts
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── media/                    # Media resources
```

## Installation

### Environment Setup

Create a virtual environment with Python 3.10 and activate it:

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

Install ffmpeg in your environment:

```bash
conda install ffmpeg -c conda-forge
```

### Install LeRobot

Clone the repository and install:

```bash
git clone https://github.com/Hiwonder/LeRobot.git
cd LeRobot
pip install -e .
```

For simulation environments:

```bash
pip install -e ".[aloha, pusht]"
```

## Version Information
- **Current Version**: Based on LeRobot v0.1.0
- **Python Version**: 3.10+
- **PyTorch Version**: 2.2+

### Related Technologies
- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [Hugging Face](https://huggingface.co/) - AI Model Hub
- [OpenCV](https://opencv.org/) - Computer Vision Library
- [WandB](https://wandb.ai/) - Experiment Tracking

---

**Note**: This repository is adapted from the original [Hugging Face LeRobot](https://github.com/huggingface/lerobot) for educational purposes. For detailed tutorials and documentation, please refer to the [Official LeRobot Documentation](https://huggingface.co/docs/lerobot).
