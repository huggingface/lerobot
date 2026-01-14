# Customized Lerobot Debug

## 1. Env setup

Install astral [uv](https://docs.astral.sh/uv/) and start a new env

```
uv venv --python=3.10
source .venv/bin/activate
```

Install repo:

```bash
# Check https://pytorch.org/get-started/locally/ for your system
uv pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0"  # --index-url https://download.pytorch.org/whl/cu1XX

# Flash attention dependencies
uv pip install ninja "packaging>=24.2,<26.0"

# Install flash attention (requires CUDA)
uv pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation

# Verify installation
python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"
```

Install LeRobot with Groot Dependencies

```bash
uv pip install -e ".[groot]"
```

Install Additional N1.6 Dependencies

```bash
uv pip install lmdb==1.7.5
uv pip install albumentations==1.4.18
uv pip install "faker>=33.0.0,<35.0.0"
uv pip install tyro
uv pip install "matplotlib>=3.10.3,<4.0.0"
```

## 2. Open Loop Test

Install system ffmpeg

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg

# verify
python -c "import torch, torchcodec; print('torch', torch.__version__); print('torchcodec', torchcodec.__version__)"
```


```bash
cd src/lerobot/scripts/
python open_loop_eval_v3.py \
        --dataset-repo-id=izuluaga/finish_sandwich \
        --policy-repo-id=nvkartik/gr00t_n1d6-finish_sandwich-relative-action-true-tune-30k \
        --episode-ids=10 \
        --save-dir=./outputs \
        --action-horizon=16 \
        --steps=400 \
        --inference-interval=2 \
        --action-offset=7
```

## 3. Dataset

View dataset
```
https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fizuluaga%2Ffinish_sandwich%2Fepisode_0
```

## 4. Real Robot: LeRobot SO101

See this: https://huggingface.co/docs/lerobot/so101

```
pip install -e ".[feetech]"
```

Find port
```
lerobot-find-port
```

On Linux, it is usually: '/dev/ttyACM0'

```
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \ 
    --robot.id=blackf
```

## 5. Real Robot: policy eval

```
rm -rf /home/yizzhao/.cache/huggingface/lerobot/izuluaga/eval_finish_sandwich

lerobot-record   \
  --robot.type=so101_follower   \
  --robot.id=blank_follower   \
  --robot.port=/dev/ttyACM0   \
  --robot.cameras='{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30, fourcc: MJPG}}'   \
  --dataset.repo_id=izuluaga/eval_finish_sandwich   \
  --dataset.single_task="finish the ham cheese olives sandwich"   \
  --policy.type=gr00t_n1d6   \
  --policy.repo_id=nvkartik/gr00t_n1d6-finish_sandwich-relative-action-true-tune-30k    \
  --dataset.num_episodes=10
```

modified:

rm -rf /home/yizzhao/.cache/huggingface/lerobot/izuluaga/eval_finish_sandwich

lerobot-record   \
  --robot.type=so101_follower   \
  --robot.id=blank_follower   \
  --robot.port=/dev/ttyACM0   \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}'   \
  --dataset.repo_id=izuluaga/eval_finish_sandwich   \
  --dataset.single_task="finish the ham cheese olives sandwich"   \
  --policy.type=gr00t_n1d6   \
  --policy.repo_id=nvkartik/gr00t_n1d6-finish_sandwich-relative-action-true-tune-30k    \
  --dataset.num_episodes=10