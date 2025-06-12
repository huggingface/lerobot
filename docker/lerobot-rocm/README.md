# LeRobot Docker Setup

This repository contains a Docker setup for running LeRobot with ROCm GPU support.

## Prerequisites

- Docker installed on your system
- AMD GPU with ROCm support
- ROCm drivers installed on the host system

## Building the Docker Image

Build the Docker image from the Dockerfile:

```bash
docker build -t lerobot:latest .
```

## Running the Container

### Running a bash session
```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size 16G \
  --ipc=host \
  --network=host \
  lerobot bash
```

### With Robot Hardware Access
To access robot arms and other hardware devices, pass the device paths using `--device` flags:

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --device=/dev/ttyACM0:/dev/ttyACM0 \
  --device=/dev/ttyACM1:/dev/ttyACM1 \
  --group-add video --group-add render \
  --shm-size 16G \
  --ipc=host \
  --network=host \
  lerobot bash
```

> **Note:** Robot arms typically connect via USB serial devices on Linux e.g. `/dev/ttyACM0`, `/dev/ttyACM1`, etc. Use `ls /dev/tty*` on your host to identify the correct device paths for your hardware.

## Verifying GPU Access

Once inside the container, verify ROCm is working:

```bash
# Check ROCm installation
rocm-smi

# Test PyTorch GPU access
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

## Verifying LeRobot installation

Verify the lerobot install by running one of the shipped examples

```bash
python lerobot/examples/2_evaluate_pretrained_policy.py
```

## Troubleshooting

**Permission denied for GPU devices:**
- Ensure your user is in the `video` and `render` groups on the host
- Check ROCm driver installation

**Out of shared memory errors:**
- Increase `--shm-size` value
- Reduce `num_workers` in PyTorch DataLoaders

**Robot hardware not accessible:**
- Check device permissions: `ls -la /dev/ttyACM*`
- Add your user to the `dialout` group: `sudo usermod -aG dialout $USER`
- Verify devices are connected: `lsusb` or `dmesg | grep tty`

**Using an iGPU:**
- `export HSA_OVERRIDE_GFX_VERSION=11.0.0`