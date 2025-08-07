# LeRobot — Quick-Start Guide (Linux & macOS)

This README walks you through **cloning, installing, and running** a pretrained pick-and-place policy on a SO-100 follower arm using **LeRobot**.  
Tested on Ubuntu 22.04 LTS and macOS 14+.

---

## 1  Clone the repository

```bash
git clone https://github.com/<YOUR-ORG>/lerobot.git
cd lerobot
# 2-a  Install Miniforge (Linux aarch64 example)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge
source ~/miniforge/etc/profile.d/conda.sh

# 2-b  Create & activate an environment
conda create -y -n lerobot python=3.10
conda activate lerobot

# 2-c  Core user-space packages
conda install -y ffmpeg -c conda-forge   # video I/O
pip install -e .                         # editable LeRobot

# Example Inference Command : 
python -m lerobot.record \
  --robot.type=so100_follower \
  --robot.id=so100_follow \
  --robot.port=/dev/tty.usbmodem58FD0171971 \
  --robot.cameras='{
    "gripper": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30},
    "top":     {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}
  }' \
  --display_data=false \
  --dataset.repo_id=adungus/eval_PP-v1_act40k-v1 \
  --dataset.root="./local_PP-v1_act40k-v1" \
  --dataset.single_task="Put the black box in the bowl" \
  --policy.path=adungus/PP-v1_act40k

| Flag                    | What it does                            | Typical values / notes                                                                                                                  |
| ----------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `--robot.type`          | Robot client implementation             | Always `so100_follower` for the SO-100 arm                                                                                              |
| `--robot.id`            | Friendly alias for this robot           | Any slug (e.g. `so100_follow`)                                                                                                          |
| `--robot.port`          | Serial port for the MCU                 | **macOS:** `/dev/tty.usbmodem*` • **Linux:** `/dev/ttyACM0` or `/dev/ttyACM1`<br>May need:<br>`sudo chmod 666 /dev/ttyACM0`             |
| `--robot.cameras`       | JSON describing every camera            | Maintain **index order** (0 = gripper, 1 = overhead). Realsense devices are **not yet fully supported**—use UVC webcams or CSI cameras. |
| `--display_data`        | Show live frames during rollout         | `false` for max performance; `true` for debugging                                                                                       |
| `--dataset.repo_id`     | Hugging Face repo for recorded episodes | **Must start with `eval`** during inference runs (e.g. `eval_PP-v1_*`).                                                                 |
| `--dataset.root`        | Local storage path                      | Strongly recommended—otherwise each episode is encoded & pushed (\~2-3 min/episode).                                                    |
| `--dataset.single_task` | Natural-language task label             | A clear description improves evaluation bookkeeping.                                                                                    |
| `--policy.path`         | HF repo or local checkpoint             | Any repo/checkpoint containing compatible weights.                                                                                      |

# Common Issues & Resolutions

Use if --robot.port cannot be 'found' or accessed
sudo chmod 666 /dev/ttyACM0   # on Linux
sudo chmod 666 /dev/tty.usbmodem58FD0171971 # on Mac

Make sure cameras index or path is appropriately matched. Doesn't have to be listed in order, just Gripper -> 0 & Top -> 1 if find_cameras.py in outputs orders them like this ; 
  --robot.cameras='{
    "gripper": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30},
    "top":     {"type": "opencv", "index_or_path": 1, "width": 1280, "height": 720, "fps": 30}
  }' \


