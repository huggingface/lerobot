# Using the [Koch v1.1](https://github.com/jess-moss/koch-v1-1) with LeRobot

## Table of Contents

  - [A. Order and Assemble the parts](#a-order-and-assemble-the-parts)
  - [B. Install LeRobot](#b-install-lerobot)
  - [C. Configure the Motors](#c-configure-the-motors)
  - [D. Calibrate](#d-calibrate)
  - [E. Teleoperate](#e-teleoperate)
  - [F. Record a dataset](#f-record-a-dataset)
  - [G. Visualize a dataset](#g-visualize-a-dataset)
  - [H. Replay an episode](#h-replay-an-episode)
  - [I. Train a policy](#i-train-a-policy)
  - [J. Evaluate your policy](#j-evaluate-your-policy)
  - [K. More Information](#k-more-information)

## A. Order and Assemble the parts

Follow the sourcing and assembling instructions provided on the [Koch v1.1 Github page](https://github.com/jess-moss/koch-v1-1). This will guide you through setting up both the follower and leader arms, as shown in the image below.

<div style="text-align:center;">
  <img src="../media/tutorial/koch_v1_1_leader_follower.webp?raw=true" alt="Koch v1.1 leader and follower arms" title="Koch v1.1 leader and follower arms" width="50%">
</div>

For a visual walkthrough of the assembly process, you can refer to [this video tutorial](https://youtu.be/8nQIg9BwwTk).

> [!IMPORTANT]
> Since the production of this video, we simplified the configuration phase (detailed in [section C](#c-configure-the-motors)) of the motors.
> Because of this, two things differ from the instructions in that video:
> - Don't plug all the motors cables right away and wait for being instructed to do so in [section C](#c-configure-the-motors).
> - Don't screw in the controller board (PCB) to the base right away and wait for being instructed to do so in [section C](#c-configure-the-motors).


## B. Install LeRobot

> [!TIP]
> We use the Command Prompt (cmd) quite a lot. If you are not comfortable using the cmd or want to brush up using the command line you can have a look here: [Command line crash course](https://developer.mozilla.org/en-US/docs/Learn_web_development/Getting_started/Environment_setup/Command_line)

Follow instructions on our [README](https://github.com/huggingface/lerobot) to install LeRobot.

In addition to these instructions, you need to install the dynamixel sdk:
```bash
pip install -e ".[dynamixel]"
```

## C. Configure the motors

### 1. Find the USB ports associated to each arm

For each controller board (Waveshare Serial Bus Servo Driver Board, one for the leader arm and one for the follower), connect it first to your computer through usb. To then find the internal port its connected to -which we will need later on- run the utility script:
```bash
python -m lerobot.find_port
```

> [!NOTE]
> Note: On Linux, you might need to give access to the USB ports by running:
> ```bash
> sudo chmod 666 /dev/ttyACM0
> sudo chmod 666 /dev/ttyACM1
> ```

This will first display all currently available ports on your computer. As prompted by the script, unplug the controller board usb cable from your computer. The script will then detect which port has been disconnected and will display it.


Example output when identifying the leader arm's port (e.g., `/dev/tty.usbmodem575E0031751` on Mac, or possibly `/dev/ttyACM0` on Linux):
```
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your MotorsBus and press Enter when done.

[...Disconnect leader arm and press Enter...]

The port of this MotorsBus is /dev/tty.usbmodem575E0031751
Reconnect the usb cable.
```

You can now reconnect the usb cable to your computer.

### 2. Set the motors ids and baudrate

Each motor is identified by a unique id on the bus. When brand new, motors usually come with a default id of `1`. For the communication to work properly between the motors and the controller, we first need to set a unique, different id to each motor. Additionally, the speed at which data is transmitted on the bus is determined by the baudrate. In order to talk to each other, the controller and all the motors need to be configured with the same baudrate.

To that end, we first need to connect to each motor individually with the controller in order to set these. Since we will write these parameters in the non-volatile section of the motors' internal memory (EEPROM), we'll only need to do this once.

> [!NOTE]
> Note: If you are repurposing motors from another robot, you will probably also need to perform this step as the ids and baudrate likely won't match.

Connect the usb cable from your computer and the 5V power supply to the leader arm's controller board. Then, run the following command with the port you got from the previous step. You'll also need to give your leader arm a name with the `id` parameter.

```bash
python -m lerobot.setup_motors \
    --device.type=so100_leader \
    --device.port=/dev/tty.usbmodem575E0031751 \  # <- paste here the port found at previous step
    --device.id=my_awesome_leader_arm  # <- give it a nice, unique name
```

Note that the command above is equivalent to running the following script:
<details>
<summary>Setup script</summary>

  ```python
  from lerobot.common.teleoperators.koch import KochLeader, KochLeaderConfig

  config = KochLeaderConfig(
      port="/dev/tty.usbmodem575E0031751",
      id="my_awesome_leader_arm",
  )
  leader = KochLeader(config)
  leader.setup_motors()
  ```
</details>


You should see the following instruction
```
Connect the controller board to the 'gripper' motor only and press enter.
```

As instructed, plug the gripper's motor. Make sure it's the only motor connected to the board, and that the motor itself is not yet daisy chained to any other motor. As you press `[Enter]`, the script will automatically set the id and baudrate for that motor.


<details>
<summary>Troubleshooting</summary>

  If you get an error at that point, check your cables and make sure they are plugged-in properly:
  - Power supply
  - USB cable between from your computer to the controller board
  - The 3-pin cable from the controller board to the motor.

  If you are using a Waveshare controller board, make sure that the two jumpers are set on the `B` channel (USB).
</details>

You should then see the following message:
```
'gripper' motor id set to 6
```

Followed by the next instruction:
```
Connect the controller board to the 'wrist_roll' motor only and press enter.
```

You can disconnect the 3-pin cable from the controller board but you can leave it connected to the gripper motor on the other end as it will already be in the right place. Now, plug-in another 3-pin cable to the wrist roll motor and connect it to the controller board. As with the previous motor, make sure it is the only motor connected to the board and that the motor itself isn't connected to any other one.

Repeat the operation for each motor as instructed.

> [!TIP]
> Check your cabling at each step before pressing Enter. For instance, the power supply cable is not solidly anchored to the board and might disconnect easily as you manipulate the board.

When you are done, the script will simply finish, at which point the motors are ready to be used. You can now plug the 3-pin cable from each motor to the next one, and the cable from the first motor (the 'shoulder pan' with id=1) to the controller board, which can now be attached to the base of the arm.

## D. Calibrate

Next, you'll need to calibrate your SO-100 robot to ensure that the leader and follower arms have the same position values when they are in the same physical position. This calibration is essential because it allows a neural network trained on one SO-100 robot to work on another.

#### a. Manual calibration of follower arm

> [!IMPORTANT]
> Contrarily to step 6 of the [assembly video](https://youtu.be/FioA2oeFZ5I?t=724) which illustrates the auto calibration, we will actually do manual calibration of follower for now.

You will need to move the follower arm to these positions sequentially:

| 1. Zero position                                                                                                                                             | 2. Rotated position                                                                                                                                                   | 3. Rest position                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="../media/so100/follower_zero.webp?raw=true" alt="SO-100 follower arm zero position" title="SO-100 follower arm zero position" style="width:100%;"> | <img src="../media/so100/follower_rotated.webp?raw=true" alt="SO-100 follower arm rotated position" title="SO-100 follower arm rotated position" style="width:100%;"> | <img src="../media/so100/follower_rest.webp?raw=true" alt="SO-100 follower arm rest position" title="SO-100 follower arm rest position" style="width:100%;"> |

Make sure both arms are connected and run this script to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

#### b. Manual calibration of leader arm
Follow step 6 of the [assembly video](https://youtu.be/FioA2oeFZ5I?t=724) which illustrates the manual calibration. You will need to move the leader arm to these positions sequentially:

| 1. Zero position                                                                                                                                       | 2. Rotated position                                                                                                                                             | 3. Rest position                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="../media/so100/leader_zero.webp?raw=true" alt="SO-100 leader arm zero position" title="SO-100 leader arm zero position" style="width:100%;"> | <img src="../media/so100/leader_rotated.webp?raw=true" alt="SO-100 leader arm rotated position" title="SO-100 leader arm rotated position" style="width:100%;"> | <img src="../media/so100/leader_rest.webp?raw=true" alt="SO-100 leader arm rest position" title="SO-100 leader arm rest position" style="width:100%;"> |

Run this script to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```

## E. Teleoperate

**Simple teleop**
Then you are ready to teleoperate your robot! Run this simple script (it won't connect and display the cameras):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```


#### a. Teleop with displaying cameras
Follow [this guide to setup your cameras](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#c-add-your-cameras-with-opencvcamera). Then you will be able to display the cameras on your computer while you are teleoperating by running the following code. This is useful to prepare your setup before recording your first dataset.

> **NOTE:** To visualize the data, enable `--control.display_data=true`. This streams the data using `rerun`.

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=teleoperate
```

## F. Record a dataset

Once you're familiar with teleoperation, you can record your first dataset with SO-100.

If you want to use the Hugging Face hub features for uploading your dataset and you haven't previously done it, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Store your Hugging Face repository name in a variable to run these commands:
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Record 2 episodes and upload your dataset to the hub:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/so100_test \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

Note: You can resume recording by adding `--control.resume=true`.

## G. Visualize a dataset

If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/so100_test
```

If you didn't upload with `--control.push_to_hub=false`, you can also visualize it locally with (a window can be opened in the browser `http://127.0.0.1:9090` with the visualization tool):
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/so100_test \
  --local-files-only 1
```

## H. Replay an episode

Now try to replay the first episode on your robot:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/so100_test \
  --control.episode=0
```

## I. Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so100_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so100_test \
  --job_name=act_so100_test \
  --policy.device=cuda \
  --wandb.enable=true
```

Let's explain it:
1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/so100_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor sates, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

Training should take several hours. You will find checkpoints in `outputs/train/act_so100_test/checkpoints`.

To resume training from a checkpoint, below is an example command to resume from `last` checkpoint of the `act_so100_test` policy:
```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_so100_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## J. Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_so100_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_so100_test/checkpoints/last/pretrained_model
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with  (e.g. `outputs/train/eval_act_so100_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_so100_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_so100_test`).

## K. More Information

Follow this [previous tutorial](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data) for a more in-depth tutorial on controlling real robots with LeRobot.

> [!TIP]
>  If you have any questions or need help, please reach out on [Discord](https://discord.com/invite/s3KuuzsPFb) in the channel [`#so100-arm`](https://discord.com/channels/1216765309076115607/1237741463832363039).
