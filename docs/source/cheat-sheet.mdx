# Cheat sheet

All of the LeRobot commands in one place. If you forgot how to use a specific command or want to learn about a new one you can do it here.

> [!WARNING]
> For all of the commands listed below remember to change the ports/names/ids to your own values!

> [!TIP]
> Another great way to look at all the commands and get them configured for your specific setup is to use this [Jupyter Notebook](https://github.com/huggingface/lerobot/blob/main/examples/notebooks/quickstart.ipynb).

### Setup and installation

For installation please look at [LeRobot Installation](https://huggingface.co/docs/lerobot/main/en/installation).

### Useful tools

###### Find port

Use this to identify which serial ports your robots are connected to. Follow the instructions in your terminal: you will be asked to unplug the USB cable and press Enter. The script will then detect and print the correct serial port for that robot.

```bash
lerobot-find-port
```

###### Find cameras

Quickly find camera indices and verify their output. This command prints camera information to the terminal and saves test frames from each detected camera to `lerobot/outputs/captured_images`

```bash
lerobot-find-cameras
```

### Calibration

In most cases you will need to perform calibration just once for each robot and teleoperation device. Before performing the calibration make sure that all the joints are roughly in the middle position.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm
```

Make sure that you use the same IDs used during calibration later for the other scripts. That's how LeRobot finds the calibration files.

### Teleoperation

Teleoperating with two cameras and displaying the data with Rerun.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --display_data=true
```

### Recording a dataset

The dataset is automatically uploaded to the server and saved under repo_id, make sure you are logged in to your HF account with CLI:
`hf auth login`

You can get the token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=${HF_USER}/so101_dataset_test \
    --dataset.num_episodes=30 \
    --dataset.single_task="put the red brick in a bowl" \
    --dataset.streaming_encoding=true \
    --display_data=true
```

While collecting the dataset you can control the process with your keyboard:
Control the data recording flow using keyboard shortcuts:

- Press **Right Arrow (`→`)**: Save episode and move to the next.
- Press **Left Arrow (`←`)**: Delete current episode and retry.
- Press **Escape (`ESC`)**: Stop, encode videos, and upload.

### Training

Depending on your hardware training the policy might take a few hours. That's how you train simple `ACT` policy:

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_dataset_test \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_test \
    --job_name=act_so101_test \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id=${HF_USER}/policy_test \
    --steps=20000
```

- Policy Types: `act`, `diffusion`, `smolvla`, `pi05`
- Devices: `cuda` (NVIDIA), `mps` (Apple Silicon), `cpu`

If you want to fine-tune a specific model you can provide the path to the model. In this case path is enough and type can be skipped.

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_dataset_test \
    --policy.path=username/the_policy_to_finetune \
    --policy.device=cuda \
    --policy.repo_id=${HF_USER}/policy_test \
    --output_dir=outputs/train/act_so101_test \
    --steps=20000
```

### Inference

Inference means running the trained policy/model on a robot. For that we use `lerobot-rollout`. You will need to provide a path to your policy. It can be a local path or a path to Hugging Face for example "lerobot/folding_latest". Your cameras configuration needs to match what was used when collecting the dataset. Duration is in seconds if unspecified, it will run forever.

> [!TIP]
> If you are using the previous release V0.5.1 instead of `lerobot-rollout` you need to use `lerobot-record`. More information [here](https://huggingface.co/docs/lerobot/v0.5.1/en/il_robots#run-inference-and-evaluate-your-policy).

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=${HF_USER}/my_policy \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video1, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video5, width: 640, height: 480, fps: 30}}" \
    --task="Put lego brick into the transparent box" \
    --duration=60
```
