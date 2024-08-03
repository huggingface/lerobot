# Robots in the real-world

This tutorial explains how to get started with real robots and train a neural network to control them autonomously.

It covers how to:
1. order and assemble your robot,
2. connect your robot, configure and calibrate it,
3. record your dataset and visualize it,
4. train a policy on your data and make sure it's ready for evaluation,
5. evaluate your policy and visualize the result afterwards.

Following these steps, you will reproduce behaviors like picking a lego block and placing it in a bin with a relatively high success rate.

TODO add video of the task
https://x.com/RemiCadene/status/1814680760592572934

While this tutorial is general and easily extendable to any type of robots by changing a configuration, it is based on the [Koch v1.1](https://github.com/jess-moss/koch-v1-1) affordable robot. Koch v1.1 is composed of a leader arm and a follower arm with 6 motors each. In addition, various cameras can record the scene and serve as visual sensors for the robot.

During data collection, you will control the follower arm by moving the leader arm. This is called "teleoperation". It is used to collect robot trajectories.
Then, you will train a neural network to imitate these trajectories. Finally, you will deploy your neural netowrk to autonomously control your robot.

Note: If you have issue at any step of the tutorial, ask for help on [Discord](https://discord.com/invite/s3KuuzsPFb).

## 1. Order and Assemble your Koch v1.1

Follow the sourcing an assembly instructions on the [Koch v1.1 github page](https://github.com/jess-moss/koch-v1-1).

![Koch v1.1 leader and follower arms](../media/koch/leader_follower.webp?raw=true "Koch v1.1 leader and follower arms")

## 2. Connect, Configure, and Calibrate your Koch v1.1

Connect the leader arm (the smaller one) with the 5V alimentation and the follower arm with the 12V alimentation. Then connect both arms to your computer with USB.

### Control your motors with DynamixelMotorsBus

We use [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py) to efficiently read from and write to the motors connected as a chain to the corresponding usb bus. Underneath, it relies on the python [dynamixel sdk](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

**Instantiate**

You will need to instante two [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py), one for each arm, with their corresponding usb port (e.g. `DynamixelMotorsBus(port="/dev/tty.usbmodem575E0031751"`).

To find their corresponding ports, run our utility script twice:
```bash
python lerobot/common/robot_devices/motors/dynamixel.py
```

A first time to find the port `/dev/tty.usbmodem575E0031751` of the leader arm:
```
Finding all available ports for the DynamixelMotorsBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your DynamixelMotorsBus and press Enter when done.

**Disconnect leader arm and press Enter**

The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751
Reconnect the usb cable.
```

A second time to find the port `/dev/tty.usbmodem575E0032081` of the follower arm:
```
Finding all available ports for the DynamixelMotorsBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your DynamixelMotorsBus and press Enter when done.

**Disconnect follower arm and press Enter**

The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0032081
Reconnect the usb cable.
```

Then you will need to list their motors with their name, motor index, and model.
Importantly, the initial motor index from factory for every motors is `1`. However, unique indices are required for these motors to function in a chain on a common bus. To this end, you will need to set different indices. We advise to follow the ascendant convention starting from index `1` (e.g. `1,2,3,4,5,6`). These indices will be written inside the persisting memory of each motor during the first connection.

Update the corresponding ports of this code and run it to instantiate the Koch leader and follower arms:
```python
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

leader_arm = DynamixelMotorsBus(
    port="/dev/tty.usbmodem575E0031751",
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_arm = DynamixelMotorsBus(
    port="/dev/tty.usbmodem575E0032081",
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)
```

Also, update the ports of the default yaml file for Koch robot [`lerobot/configs/robot/koch.yaml`](../lerobot/configs/robot/koch.yaml). It is used to instantiate a robot in all our scripts. We will explain how this works later on.

**Configure and Connect**

Then, you will need to configure your motors. During the first connection of the motors, [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py) automatically detects a mismatch between the present motor indices (all `1` by factory default) and your specified motor indices (e.g. `1,2,3,4,5,6`). This triggers the configuration procedure which requires to unplug the power cord and motors, and to sequentially plug each motor again, starting from the closest to the bus.

Take a look at this youtube video for help: TODO(rcadene)

Run this in the same python session in your terminal to connect and configure the leader arm:
```python
leader_arm.connect()
```

Here is an example of connecting the leader arm for the first time:
```
Read failed due to communication error on port /dev/tty.usbmodem575E0032081 for group_key ID_shoulder_pan_shoulder_lift_elbow_flex_wrist_flex_wrist_roll_gripper: [TxRxResult] There is no status packet!

/!\ A configuration issue has been detected with your motors:
- Verify that all the cables are connected the proper way. Before making a modification, unplug the power cord to not damage the motors. Rewire correctly. Then plug the power again and relaunch the script.
- If it's the first time that you use these motors, press Enter to configure your motors...

Motor indices detected: {9600: [1]}

1. Unplug the power cord
2. Plug/unplug minimal number of cables to only have the first 1 motor(s) (['shoulder_pan']) connected.
3. Re-plug the power cord
Press Enter to continue...

[...] *Run all the procedure*

Setting expected motor indices: [1, 2, 3, 4, 5, 6]
```

Now do the same for the follower arm:
```python
follower_arm.connect()
```

Congrats, now both arms are well configured and connected! You won't have to follow the configuration procedure ever again!

Note: If the configuration didn't work, you might need to update the firmware using [DynamixelWizzard2](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2). You might also need to manually configure the motors. Similarly, you will need to connect each motor seperately to the bus. You will need to set correct indices and set their baudrates to `1000000`. Take a look at this youtube video for help: https://www.youtube.com/watch?v=JRRZW_l1V-U


**Read and Write**

Just to get familiar with how `DynamixelMotorsBus` is used to command the motors, let's try to read from them:
```python
position = leader_arm.read("Present_Position")
print(position)
>>> array([2054,  523, 3071, 1831, 3049, 2441], dtype=int32)

position = follower_arm.read("Present_Position")
print(position)
>>> array([2003, 1601,   56, 2152, 3101, 2283], dtype=int32)
```

Try to move the arms in various positions and see how if affects the values.

Now let's try to enable torque in the follower arm:
```python
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)
```

The follower arm should be stuck in its current position. Don't try to manually move it while torque is enabled as it might damage the motors. Instead try to move it using the code:
```python
# Get the current position
position = follower_arm.read("Present_Position")

# Update first motor (shoulder_pan) position by +10 steps
position[0] += 10
follower_arm.write("Goal_Position", position)

# Update all motors position by -30 steps
position -= 30
follower_arm.write("Goal_Position", position)

# Update gripper by +30 steps
position[-1] += 30
follower_arm.write("Goal_Position", position[-1], "gripper")
```

When you are done, disable the torque by running:
```python
# Warning: hold your robot so that it doesn't fall
follower_arm.write("Torque_Enable", TorqueMode.DISABLED.value)
```

And disconnect the two arms:
```python
leader_arm.disconnect()
follower_arm.disconnect()
```

You can also unplug the power cord which will disable torque and disconnect.

### Teleoperate your Koch v1.1 with KochRobot

**Instantiate**

```python
from lerobot.common.robot_devices.robots.koch import KochRobot

robot = KochRobot(
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_path=".cache/calibration/koch.pkl",
)
```

**Calibrate and Connect**

```python
robot.connect()

>>>
```

```python
degrees = robot.leader_arms["main"].read("Present_Position")
print(degrees)
>>> array([ -0.43945312, 133.94531   , 179.82422   , -18.984375  ,
        -1.9335938 ,  34.541016  ], dtype=float32)
```


```python
degrees = robot.follower_arms["main"].read("Present_Position")
print(degrees)
>>> array([ -1.0546875, 128.67188  , 174.90234  ,  -7.998047 ,  -5.4492188,
        32.34375  ], dtype=float32)
```

**Teleoperate**

TODO: explain in pseudo code what the teleop is doing

```python
import tqdm
# Teleoperate for 30 seconds since the arms are communicating at a frequency of 200Hz
for _ in tqdm.tqdm(range(30*200)):
    robot.teleop_step()
```

TODO: explain in pseudo code what the teleop(record_data=True) is doing

```python
observation, action = robot.teleop_step(record_data=True)
print(observation)
>>> {'observation.state': tensor([  7.8223, 131.1328, 165.5859, -23.4668,  -0.9668,  32.4316])}
print(action)
>>> {'action': tensor([  3.4277, 134.1211, 179.8242, -18.5449,  -1.5820,  34.7168])}
```

### Add your cameras with OpenCVCamera

**Instantiate**

The `OpenCVCamera` class allows to efficiently record images from cameras. It relies on opencv2 to communicate with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

An `OpenCVCamera` instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

To find the camera indices of your cameras, you can run our utility script that will save a few frames for each camera:
```bash
python lerobot/common/robot_devices/cameras/opencv.py --images-dir outputs/images_from_opencv_cameras
>>> TODO
```

When an `OpenCVCamera` is instantiated, if no specific config is provided, the default fps, width, height and color_mode of the given camera will be used.

Example of usage of the class:
```python
camera = OpenCVCamera(camera_index=0)
camera.connect()
color_image = camera.read()
# when done using the camera, consider disconnecting
camera.disconnect()
```

**Add to robot**

TODO: explain that the cameras run asynchronously.

```python
del robot
robot = KochRobot(
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_path=".cache/calibration/koch.pkl",
    cameras={
        "laptop": OpenCVCamera(0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(1, fps=30, width=640, height=480),
    },
)

robot.connect()

observation, action = robot.teleop_step(record_data=True)
print(observation)
>>>
print(action)
>>>
```

### Use `koch.yaml` and our `teleoperate` function

See: `lerobot/configs/robot/koch.yaml`

```bash
python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml

>>>
```

```bash
python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml \
  --robot-overrides \
    leader_arms.main.port=/dev/tty.usbmodem575E0031751 \
    follower_arms.main.port=/dev/tty.usbmodem575E0032081

>>>
```

```bash
python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml \
  --robot-overrides \
    leader_arms.main.port=/dev/tty.usbmodem575E0031751 \
    follower_arms.main.port=/dev/tty.usbmodem575E0032081
    '~cameras'
```


## 3. Record your Dataset and Visualize it

TODO: ideally we could only do this

```python
from lerobot.scripts.control_robot import busy_wait

fps = 30
record_time_s = 60
for _ in range(fps * record_time_s):
    start_time = time.perf_counter()

    observation, action = robot.teleop_step(record_data=True)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)
```

### Use `koch.yaml` and the `record` function

TODO: We added ways to write the frames to disk in multiple thread
We added warmap, reset time between episodes
At the end we encode the frames into videos
control
if fail, re-record episode
checkpointing
We consolidate the data into a LeRobotDataset and upload on the hub.

Here is an example for 1 episode
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root /tmp/data \
    --repo-id $USER/koch_test \
    --num-episodes 10 \
    --run-compute-stats 1
```

TODO: USER HF, make sure you can push

### Replay episode on your robot with the `replay` function

```bash
python lerobot/scripts/control_robot.py replay \
    --fps 30 \
    --root /tmp/data \
    --repo-id $USER/koch_test \
    --episode 0
```

Note: TODO
```bash
export DATA_DIR=data
```

### Visualize all episodes

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test
```


## 4. Train a policy on your data

### Use our `train` script

```bash
python lerobot/scripts/train.py \
    policy=act_koch_real \
    env=koch_real \
    dataset_repo_id=$USER/koch_pick_place_lego \
    hydra.run.dir=outputs/train/act_koch_real
```

TODO: image and plots of wandb

```bash
ckpt=100000
huggingface-cli upload cadene/2024_07_27_act_koch_pick_place_1_lego_raph_nightly_${ckpt} \
  outputs/train/2024_07_27_act_koch_pick_place_1_lego_raph_nightly/checkpoints/${ckpt}/pretrained_model
```

### Visualize predictions on training set

```bash
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/koch_pick_place_1_lego \
    --episodes 0 1 2 \
    -p ../lerobot/outputs/train/2024_07_29_act_koch_pick_place_1_lego_mps/checkpoints/006000/pretrained_model
```

## 5. Evaluate your policy

### Use our `record` function

```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root /tmp/data \
    --repo-id $USER/eval_koch_test \
    --num-episodes 10 \
    --run-compute-stats 1
    -p ../lerobot/outputs/train/2024_07_29_act_koch_pick_place_1_lego_mps/checkpoints/006000/pretrained_model
```

### Visualize evaluation afterwards

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test
```


## What's next?

### More datasets

Collect a slightly more difficult dataset, like grasping 5 lego blocks in a row, and co-train on it

###


- Improve the dataset









Prices

 Some parts and prices from the bill of materials might differ with respect to the geo location ; but we made sure it works.


via the high frequency communication of goal positions to the follower arm.

to predict the future actions given the state and camera frames as input ; and deploy it to autonomously control the robot via the high frequency communication of goal positions to the follower arm.


More specifically, the present position of the motors of the leader arm is read at high frequency and sent as a goal position for the motors of the follower arm, which effectively "follow" the movements of the leader arm. While you teleoperate the robot, a few modalities are recorded:
- the present position of the follower arm, called the "state",
- the goal position sent to the follower arm, called the "action",
- the video stream from the cameras.

Once the parts are received, follow this video to guide you through the assembly: TODO
