# Distributed Real-Time Chunking (DRTC)

Distributed Real-Time Chunking (DRTC) is an async inference approach for action chunking policies in distributed client-server deployments. It combines RTC-compatible in-painting with resilient message handling under unreliable communication.

## Abstract

Action chunking policies are increasingly run on remote servers due to model size, hardware constraints on edge devices, and cost constraints. Async inference has become a common strategy for enabling smooth action trajectories and closing the gap between action chunks due to inference and network latency. Existing approaches such as Real-Time Execution of Action Chunking Flow Policies~\cite{black2025rtc}, SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics~\cite{shukor2025smolvla} address different problems within async inference. RTC in-painting focuses on the problem of trajectory discontinuities and mode switching, while SmolVLA Async Inference addresses a distributed-client server architecture. To our knowledge, what is currently missing from the literature is a unified async inference approach that combines RTC in-painting, a well defined distributed architecture, and resilient behavior under unreliable communication channels.
We present Distributed Real Time Chunking (DRTC), an RTC compatible approach for distributed client-server scenarios that is designed to handle these failure modes, and we evaluate DRTC's behavior under injected faults. A cooldown mechanism enables recovery from lost and delayed messages. Thread message passing and action schedule merging are modeled as Last Write Wins (LWW) registers from the CRDT~\cite{shapiro2011crdt} literature, and described analytically by the semilattice join. The semilattice join operation absorbs reordered and duplicated messages and ensures monotone data-flow from observation to action execution.

Full technical blog: https://jackvial.com/posts/distributed-real-time-chunking.html

The implementation presented here is built on LeRobot, although the approach in general is library and language agnostic.

## Prerequisites

- **uv** (Python package/project manager): https://docs.astral.sh/uv/getting-started/installation/
- Prime Intellect account and prime CLI installed locally: https://www.primeintellect.ai/
  - `~/.prime/config.json` with your API key and SSH key path (see below)
- Tailscale account/network for secure connectivity between client and remote server: https://tailscale.com/
- SO101 robot setup (default tested hardware profile in this repo)

## Getting Started

### 0. Set Up the Local Environment

Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment with Python 3.12 and activate it:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install the project with the required extras (`smolvla`, `async`, `feetech`):

```bash
uv pip install -e ".[smolvla,async,feetech]"
```

### 1. Provision A Remote Policy Server On Prime Intellect

Run from this repository root:

```bash
./scripts/provision_prime_lerobot.sh
```

This script searches for available GPUs with the required CUDA image, presents them in an interactive table for you to choose from, provisions the selected instance, clones the repo, installs dependencies, sets up Tailscale, and prints:
- SSH connection details (`user@host` and port)
- Tailscale domain for the remote machine

To resume setup on an existing pod (e.g. after a network interruption):

```bash
./scripts/provision_prime_lerobot.sh --pod-id <POD_ID>
```

### 2. Start the policy server on Prime Intellect

SSH to the provisioned machine (use the values printed by provisioning), then start the policy server:

```bash
ssh -i <SSH_KEY_PATH> -p <SSH_PORT> <SSH_USER>@<SSH_HOST>
cd /workspace/drtc
./scripts/start_drtc_server.sh
```

Leave this process running while the client connects.

### 3. Start a local client

From your local client/robot machine, start the tunnel + client flow:

```bash
TUNNEL_SSH_PORT=<SSH_PORT> ./scripts/start_drtc_client.sh --tunnel-ssh-user-host <SSH_USER>@<SSH_HOST>
```

If your remote SSH port is the default used by the script, you can omit `TUNNEL_SSH_PORT`.

> Note: You can also run the policy server locally if preferred:
> `./scripts/start_drtc_server.sh`

## Customization (Model, Cameras, Robot)

Current repo defaults remain preconfigured for your existing setup:
- Robot type: `so101`
- Model: `jackvial/so101_smolvla_pickplaceorangecube_e100`
- Camera paths/format from the current SO101 setup

`robot_type` is configurable (currently supported: `so101`/`so101_follower`, `so100`/`so100_follower`).

### Local client path overrides

When using `examples/tutorial/async-inf/robot_client_drtc.py`, you can override settings via environment variables:

```bash
export LEROBOT_POLICY_TYPE=smolvla
export LEROBOT_PRETRAINED_NAME_OR_PATH=your-org/your-model
export LEROBOT_ROBOT_TYPE=so101
export LEROBOT_FOLLOWER_PORT=/dev/ttyACM1
export LEROBOT_FOLLOWER_ID=your_robot_id
export LEROBOT_CAMERA1_PATH=/dev/v4l/by-path/your-camera-1
export LEROBOT_CAMERA2_PATH=/dev/v4l/by-path/your-camera-2
export LEROBOT_CAMERA_WIDTH=800
export LEROBOT_CAMERA_HEIGHT=600
export LEROBOT_CAMERA_FPS=30
export LEROBOT_CAMERA_FOURCC=MJPG
```

### Experiment path overrides (YAML)

When running via `scripts/run_drtc_experiment_with_remote_server.sh`, set values in your experiment config YAML (loaded by `examples/experiments/run_drtc_experiment.py`):

```yaml
robot_type: so101
robot_port: /dev/ttyACM0
robot_id: so101_follower_2026_01_03

policy_type: smolvla
pretrained_name_or_path: jackvial/so101_smolvla_pickplaceorangecube_e100

camera1_path: /dev/v4l/by-path/platform-xhci-hcd.1-usb-0:2:1.0-video-index0
camera2_path: /dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0
camera_width: 800
camera_height: 600
camera_fps: 30
camera_fourcc: MJPG
camera_use_threaded_async_read: true
camera_allow_stale_frames: true
```

## Run Experiments from the Client

Use the remote experiment runner script and point it at the remote policy server host/domain:

```bash
./scripts/run_drtc_experiment_with_remote_server.sh \
  --remote-server-host <TAILSCALE_DOMAIN_OR_IP> \
  --config mixture_of_faults \
  --output_dir results/experiments
```

Another example:

```bash
./scripts/run_drtc_experiment_with_remote_server.sh \
  --remote-server-host <TAILSCALE_DOMAIN_OR_IP> \
  --config spike \
  --output_dir results/experiments
```

## Plot Results

After experiments finish, generate plots with:

```bash
uv run python examples/experiments/plot_results.py \
  --input results/experiments \
  --output results/experiments/summary
```

For a single run:

```bash
uv run python examples/experiments/plot_results.py \
  --input results/experiments/<run_name>/<run_name>.csv \
  --mode detailed \
  --output results/experiments/<run_name>/detailed
```

## Citation

If you use DRTC in your research, please cite:

```bibtex
@misc{vialdrtc2026,
  title        = {Distributed Real-Time Chunking},
  author       = {Vial, Jack},
  year         = {2026},
  howpublished = {\url{https://jackvial.com/posts/distributed-real-time-chunking.html}},
  note         = {Technical report / blog}
}
```

## Acknowledgements

The DRTC implementation presented here is build on [LeRobot](https://github.com/huggingface/lerobot) and the LeRobot RTC implementation. Special thanks to the LeRobot team and open source contributers [Eugene Mironov](https://github.com/helper2424), and [Khalil Meftah](https://github.com/s1lent4gnt) for their work on the RTC implementation.
