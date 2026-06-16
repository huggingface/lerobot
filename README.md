# LeRobot

English | [中文](https://github.com/Hiwodner-official/LeRobot/blob/main/README_cn.md)

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

### Hiwonder
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

### Imitation Learning Policies
- **ACT** - Action Chunking with Transformers
- **Diffusion Policy** - Diffusion-based action generation
- **PI0 / PI0-Fast** - Flow matching policies
- **SmolVLA** - Small vision-language-action model

### Robot Platforms
- **SO-101** - Affordable 6-DOF robotic arm
- **HopeJR** - Humanoid robot arm and hand
- **LeKiwi** - Mobile robot platform
- **SO-ARM101** - Hiwonder high-torque robotic arm

### Motor Support
- **Feetech STS/SMS series** - Serial bus servos
- **Dynamixel** - High-performance servos
- **Hiwonder HX-30HM** - 30kg magnetic-encoder servo

### Programming Interface
- **Python SDK** - Complete Python programming interface
- **PyTorch** - Deep learning framework
- **Hugging Face Hub** - Dataset and model sharing
- **WandB** - Experiment tracking and visualization

## Project Structure

```
lerobot/
├── src/lerobot/
│   ├── cameras/          # Camera drivers
│   ├── configs/          # Configuration files
│   ├── datasets/         # Dataset handling
│   ├── envs/             # Simulation environments
│   ├── model/            # Neural network models
│   ├── motors/           # Motor control drivers
│   │   ├── feetech/      # Feetech STS/SMS servos
│   │   ├── hiwonder/     # Hiwonder HX-30HM servos
│   │   └── dynamixel/    # Dynamixel servos
│   ├── policies/         # Policy implementations
│   ├── robots/           # Robot configurations
│   ├── scripts/          # Utility scripts
│   └── teleoperators/    # Teleoperation systems
├── tests/                # Unit tests
└── sources/              # Media resources
```

## Installation

### Requirements
- Python 3.12+
- PyTorch 2.7+

### Install with uv (recommended)

```bash
git clone https://github.com/Hiwodner-official/LeRobot.git
cd LeRobot
uv sync
```

### Install with pip

```bash
git clone https://github.com/Hiwodner-official/LeRobot.git
cd LeRobot
pip install -e .
```

### Install with Hiwonder servo support

```bash
pip install -e ".[hiwonder]"
```

### Install with Feetech servo support

```bash
pip install -e ".[feetech]"
```

## Version Information
- **Current Version**: v0.5.1 (based on upstream LeRobot v0.5.1)
- **Python Version**: 3.12+
- **PyTorch Version**: 2.7+

---

**Note**: This repository is adapted from the original [Hugging Face LeRobot](https://github.com/huggingface/lerobot) for use with Hiwonder hardware. For detailed tutorials and documentation, please refer to the [Official LeRobot Documentation](https://huggingface.co/docs/lerobot).
