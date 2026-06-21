## SO-ARM101 Pro Setup Notes

Leader arm:

- Port: `/dev/tty.usbmodem5B7B0098471`
- Type: `so101_leader`
- Power: 5V supply only
- Servos: 7.4V-class leader servos

Follower arm:

- Port: `/dev/tty.usbmodem5B7B0145651`
- Type: `so101_follower`
- Power: 12V supply only
- Servos: 12V follower servos

Camera:

- Overhead USB camera index: `0`

## Recalibrate Leader Arm

Use this if the leader calibration needs to be redone:

```bash
cd ~/lerobot
conda activate lerobot

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B7B0098471 \
  --teleop.id=my_leader_arm
```

If prompted:

```text
Press ENTER to use provided calibration file..., or type 'c' and press ENTER to run calibration:
```

Type:

```text
c
```

Middle pose means every joint is away from its hard stops:

- Base pointing forward.
- Shoulder roughly mid-raised.
- Elbow about 90 degrees.
- Wrist roughly straight.
- Gripper half-open.

Then move all requested joints slowly through their full usable ranges.

## Recalibrate Follower Arm

Use this if the follower calibration needs to be redone:

```bash
cd ~/lerobot
conda activate lerobot

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B7B0145651 \
  --robot.id=my_follower_arm
```

If prompted to use the existing calibration, type `c` and press Enter.

Before calibrating:

- Use the 12V supply on the follower only.
- Make sure all 6 follower servos are connected.
- Put the follower in a neutral middle pose.
- Keep one hand near power in case anything moves unexpectedly.

## Teleoperate

Use this to manually control the follower with the leader:

```bash
cd ~/lerobot
conda activate lerobot

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B7B0145651 \
  --robot.id=my_follower_arm \
  --robot.max_relative_target=10 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B7B0098471 \
  --teleop.id=my_leader_arm
```

If the arm feels too slow, increase `--robot.max_relative_target` gradually.
If it jerks or makes big corrections, lower it.

## Record Dataset

Use this to record 5 overhead-camera demos:

```bash
cd ~/lerobot
conda activate lerobot

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B7B0145651 \
  --robot.id=my_follower_arm \
  --robot.max_relative_target=15 \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B7B0098471 \
  --teleop.id=my_leader_arm \
  --dataset.repo_id=ethan/so101_overhead_test \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=0 \
  --dataset.reset_time_s=0 \
  --dataset.single_task="Pick up the object and place it in the bin" \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
```

To speed up the robot during recording:

```bash
--robot.max_relative_target=20
```

Start at `10`, then try `20` if motion is stable. If the arm jerks, overshoots, or makes large corrections, lower it again.

To speed up video saving after each episode:

```bash
--dataset.encoder_threads=4
```

Use `2` if the Mac gets sluggish or recording drops frames. Keep `--display_data=false` for faster recording.

For real training data, change:

```bash
--dataset.repo_id=ethan/so101_pickup
--dataset.num_episodes=50
```

## Recording Controls

During a task episode:

- Right arrow: finish and save the current episode.
- Left arrow: discard and rerecord the current episode.
- Escape: stop the whole recording session.

Local browser controls have been added locally. When recording starts, the browser should open automatically at:

```text
http://127.0.0.1:8765
```

If the browser does not open automatically, open that URL manually and use:

- Spacebar: same as Finish Episode. Use this for the normal record/reset rhythm.
- Finish Episode: finish and save the current episode.
- Rerecord Episode: discard and redo the current episode.
- Stop Recording: stop the whole recording session.

During reset:

- Put the object and bin back in their starting positions.
- Put the robot/leader in a good starting pose if needed.
- Press Right arrow or click Finish Episode again to finish reset and start the next episode.

Recording rhythm:

```text
Record task -> Space
Reset scene -> Space
Record next task -> Space
Reset scene -> Space
```

## Train Policy

Install training dependencies once:

```bash
cd ~/lerobot
conda activate lerobot

/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python -m pip install -e ".[feetech,dataset,viz,training]"
```

Find the latest local dataset:

```bash
ls -td ~/.cache/huggingface/lerobot/ethan/so101_overhead_test_* | head -1
```

The folder name maps to the dataset repo id. For example:

```text
~/.cache/huggingface/lerobot/ethan/so101_overhead_test_20260620_195133
```

means:

```text
ethan/so101_overhead_test_20260620_195133
```

Train a quick test policy on Mac:

```bash
cd ~/lerobot
conda activate lerobot

lerobot-train \
  --dataset.repo_id=ethan/so101_overhead_test_20260620_202117 \
  --policy.type=act \
  --policy.device=mps \
  --output_dir=outputs/train/act_so101_overhead_test \
  --job_name=act_so101_overhead_test \
  --steps=20000 \
  --batch_size=8 \
  --log_freq=50 \
  --save_freq=1000 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

Replace `ethan/so101_overhead_test_20260620_195133` with the latest dataset id from the `ls -td` command.

For a real 30-50 episode dataset, increase training:

```bash
--steps=20000
--batch_size=8
```

## Run Trained Policy

Run the trained ACT policy on the follower:

```bash
cd ~/lerobot
conda activate lerobot

lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_so101_overhead_test/checkpoints/last/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B7B0145651 \
  --robot.id=my_follower_arm \
  --robot.max_relative_target=10 \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the object and place it in the bin" \
  --duration=20 \
  --fps=30 \
  --display_data=true
```

Keep one hand near follower power during rollout. If motion is jerky or wrong, stop power immediately and lower `--robot.max_relative_target`.

## Camera Preview

Find cameras:

```bash
lerobot-find-cameras opencv
```

Camera preview images are saved under:

```text
~/lerobot/outputs/captured_images
```

Open them with:

```bash
open outputs/captured_images
```
