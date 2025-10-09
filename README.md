<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/lerobot-logo-thumbnail.png" width="100%">
  <br/>
  <br/>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/Discord-Logo-Blurple.png" alt="Discord" width="90"/>](https://discord.gg/s3KuuzsPFb)

<!-- [![Coverage](https://codecov.io/gh/huggingface/lerobot/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/huggingface/lerobot) -->

</div>

<h3 align="center">
    <p>LeRobot: State-of-the-art AI for real-world robotics</p>
</h3>

---

🤗 LeRobot acts as the unified framework for state-of-the-art robot learning, covering simulation, real-world control, datasets, and pretrained policies built in PyTorch.

It centralizes the definition of robot environments, datasets, and models, and ensure compatibility across the robotics ecosystem: if a robot or environment is supported in LeRobot, it works seamlessly with the majority of training frameworks and policy architectures (ACT, Diffusion Policy, TDMPC, SmolVLA, ...).

We aim to democratize robotics by making modern robot learning simple, modular, and open , empowering anyone to collect data, train models, and deploy them on affordable real-world robots.

There are already thousands of pretrained LeRobot models and datasets available on the Hugging Face Hub that you can explore and use to get started right away.

#### Examples of pretrained models on simulation environments

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/aloha_act.gif" width="100%" alt="ACT policy on ALOHA env"/></td>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/simxarm_tdmpc.gif" width="100%" alt="TDMPC policy on SimXArm env"/></td>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/pusht_diffusion.gif" width="100%" alt="Diffusion policy on PushT env"/></td>
  </tr>
  <tr>
    <td align="center">ACT policy on ALOHA env</td>
    <td align="center">TDMPC policy on SimXArm env</td>
    <td align="center">Diffusion policy on PushT env</td>
  </tr>
</table>

## Installation

LeRobot works with Python 3.10+ and PyTorch 2.2+.

### Environment Setup

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

When using `miniconda`, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>
> - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>
> ```bash
> conda install ffmpeg=7.1.1 -c conda-forge
> ```
>
> - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

### Install LeRobot 🤗

#### From Source

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Then, install the library in editable mode. This is useful if you plan to contribute to the code.

```bash
pip install -e .
```

> **NOTE:** If you encounter build errors, you may need to install additional dependencies (`cmake`, `build-essential`, and `ffmpeg libs`). On Linux, run:
> `sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev`. For other systems, see: [Compiling PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

For simulations, 🤗 LeRobot comes with gymnasium environments that can be installed as extras:

- [aloha](https://github.com/huggingface/gym-aloha)
- [libero](https://huggingface.co/docs/lerobot/en/libero)
- [metaworld](https://github.com/huggingface/gym-pusht)

For instance, to install 🤗 LeRobot with libero and metaworld, use:

```bash
pip install -e ".[libero, metaworld]"
```

### Installation from PyPI

**Core Library:**
Install the base package with:

```bash
pip install lerobot
```

_This installs only the default dependencies._

**Extra Features:**
To install additional functionality, use one of the following:

```bash
pip install 'lerobot[all]'          # All available features
pip install 'lerobot[aloha,pusht]'  # Specific features (Aloha & Pusht Environments)
pip install 'lerobot[feetech]'      # Feetech motor support
```

_Replace `[...]` with your desired features._

**Available Tags:**
For a full list of optional dependencies, see:
https://pypi.org/project/lerobot/

<h2>Our Robots</h2>

<details>
  <summary><b>SO-101</b> - Affordable dual-arm robot (€114/arm)</summary>
  <p align="center">
    <img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/so101/so101.webp" alt="SO-101 robot arm" width="70%"/>
  </p>
  <p align="center">
    The SO-101 is a low-cost, open-source dual-arm robot designed for imitation learning and real-world control.<br/>
    <a href="https://huggingface.co/docs/lerobot/so101">🔗 See SO-101 tutorial</a>
  </p>
</details>

<details>
  <summary><b>LeKiwi</b> - Mobile base for SO-101</summary>
  <p align="center">
    <img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/lekiwi/kiwi.webp" alt="LeKiwi mobile robot" width="70%"/>
  </p>
  <p align="center">
    LeKiwi turns SO-101 into a mobile robot, combining locomotion and manipulation for autonomous tasks.<br/>
    <a href="https://huggingface.co/docs/lerobot/lekiwi">🔗 See LeKiwi tutorial</a>
  </p>
</details>

<details>
  <summary><b>Hope JR</b> - Humanoid arm for dexterous manipulation</summary>
  <p align="center">
    <img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/hope_jr/hopejr.png" alt="HopeJR robot" width="70%"/>
  </p>
  <p align="center">
    Hope JR is a humanoid hand and arm for fine manipulation, controllable via exoskeletons and gloves.<br/>
    <a href="https://huggingface.co/docs/lerobot/hope_jr">🔗 See HopeJR tutorial</a>
  </p>
</details>

<details>
  <summary><b>SO-100</b> - Early prototype of SO-101</summary>
  <p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/so100-lerobot.jpg" alt="SO-100 robot" width="70%"/>
  </p>
  <p align="center">
    The SO-100 is the predecessor to SO-101, built for rapid prototyping and early policy testing.<br/>
    <a href="https://huggingface.co/docs/lerobot/so100">🔗 See SO-100 tutorial</a>
  </p>
</details>

<details>
  <summary><b>Koch v1.1</b> - Lightweight research manipulator</summary>
  <p align="center">
    <img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/koch/koch_v1_1.webp" alt="Koch v1.1 robot" width="70%"/>
  </p>
  <p align="center">
    Koch v1.1 is a compact research manipulator built for modularity, speed, and reproducibility.<br/>
    <a href="https://huggingface.co/docs/lerobot/koch_v1_1">🔗 See Koch v1.1 tutorial</a>
  </p>
</details>

<details>
  <summary><b>Reachy 2</b> - Open-source humanoid platform</summary>
  <p align="center">
    <img src="https://www.pollen-robotics.com/wp-content/uploads/2024/10/reachy2_full_robot.webp" alt="Reachy 2 robot" width="70%"/>
  </p>
  <p align="center">
    Reachy 2 is a community-built open humanoid, ideal for perception and embodied AI research.<br/>
    <a href="https://huggingface.co/docs/lerobot/reachy2">🔗 See Reachy 2 tutorial</a>
  </p>
</details>

<h2>LeRobotDataset</h2>
<p>
  To learn more about our <code>LeRobotDataset</code> and how to visualize it, see
  <a href="https://huggingface.co/docs/lerobot/lerobot-dataset-v3">this guide</a>.
</p>

<p>
  The <code>LeRobotDataset</code> is quickly becoming the foundation for open robotics data — unifying formats,
  tooling, and best practices across simulation and real-world datasets.
</p>



## Contribute

If you would like to contribute to 🤗 LeRobot, please check out our [contribution guide](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md).

### Add a pretrained policy

Once you have trained a policy you may upload it to the Hugging Face hub using a hub id that looks like `${hf_user}/${repo_name}` (e.g. [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)).

You first need to find the checkpoint folder located inside your experiment directory (e.g. `outputs/train/2024-05-05/20-21-12_aloha_act_default/checkpoints/002500`). Within that there is a `pretrained_model` directory which should contain:

- `config.json`: A serialized version of the policy configuration (following the policy's dataclass config).
- `model.safetensors`: A set of `torch.nn.Module` parameters, saved in [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) format.
- `train_config.json`: A consolidated configuration containing all parameters used for training. The policy configuration should match `config.json` exactly. This is useful for anyone who wants to evaluate your policy or for reproducibility.

To upload these to the hub, run the following:

```bash
huggingface-cli upload ${hf_user}/${repo_name} path/to/pretrained_model
```

See [eval.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/eval.py) for an example of how other people may use your policy.

### Acknowledgment

- The LeRobot team 🤗 for building SmolVLA [Paper](https://arxiv.org/abs/2506.01844), [Blog](https://huggingface.co/blog/smolvla).
- Thanks to Tony Zhao, Zipeng Fu and colleagues for open sourcing ACT policy, ALOHA environments and datasets. Ours are adapted from [ALOHA](https://tonyzhaozh.github.io/aloha) and [Mobile ALOHA](https://mobile-aloha.github.io).
- Thanks to Cheng Chi, Zhenjia Xu and colleagues for open sourcing Diffusion policy, Pusht environment and datasets, as well as UMI datasets. Ours are adapted from [Diffusion Policy](https://diffusion-policy.cs.columbia.edu) and [UMI Gripper](https://umi-gripper.github.io).
- Thanks to Nicklas Hansen, Yunhai Feng and colleagues for open sourcing TDMPC policy, Simxarm environments and datasets. Ours are adapted from [TDMPC](https://github.com/nicklashansen/tdmpc) and [FOWM](https://www.yunhaifeng.com/FOWM).
- Thanks to Antonio Loquercio and Ashish Kumar for their early support.
- Thanks to [Seungjae (Jay) Lee](https://sjlee.cc/), [Mahi Shafiullah](https://mahis.life/) and colleagues for open sourcing [VQ-BeT](https://sjlee.cc/vq-bet/) policy and helping us adapt the codebase to our repository. The policy is adapted from [VQ-BeT repo](https://github.com/jayLEE0301/vq_bet_official).

## Citation

If you want, you can cite this work with:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=huggingface/lerobot&type=Timeline)](https://star-history.com/#huggingface/lerobot&Timeline)

```

```
