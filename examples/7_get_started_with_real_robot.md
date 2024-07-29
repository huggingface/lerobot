# Robots in the real-world

This tutorial explains how to get started with real robots and train a neural network to control them autonomously.

It covers how to:
1. order and assemble your robot,
2. connect your robot, configure it and calibrate it,
3. record your dataset and visualize it,
4. train a policy on your data and make sure it's ready for evaluation,
5. evaluate your policy and visualize the result afterwards.

Following these steps, you should be able to reproduce behaviors like picking a lego block and placing it in a bin with a relatively high success rate.

While this tutorial is general and easily extendable to any type of robots by changing a configuration, it is based on the [Koch v1.1](https://github.com/jess-moss/koch-v1-1) affordable robot. Koch v1.1 is composed of a leader arm and a follower arm with 6 motors each. In addition, various cameras can record the scene and serve as visual sensors for the robot.

During data collection, you will control the follower arm by moving the leader arm. This is called "teleoperation". More specifically, the present position of the motors of the leader arm is read at high frequency and sent as a goal position for the motors of the follower arm, which effectively "follow" the movements of the leader arm. While you teleoperate the robot, a few modalities are recorded:
- the present position of the follower arm, called the "state",
- the goal position sent to the follower arm, called the "action",
- the video stream from the cameras.

Finally, you will train a neural network to predict the future actions given the state and camera frames as input ; and deploy it to autonomously control the robot via the high frequency communication of goal positions to the follower arm.


## 1. Order and Assemble your Koch v1.1

Follow the bill of materials on the [Koch v1.1 github page](https://github.com/jess-moss/koch-v1-1) to order a leader and a follower arm. Some parts and prices are a bit different with respect to the geo location.

Once the parts are received, follow this video to guide you through the assembly:

## 2. Connect, Configure, and Calibrate your Koch v1.1

Connect the leader arm (the smaller one) with the 5V alimentation and the follower arm with the 12V alimentation. Then connect both arms to your computer with USB.

### Control your motors with DynamixelMotorsBus

[`DynamixelMotorsBus`](lerobot/common/robot_devices/motors/dynamixel.py) allows to efficiently read from and write to the motors connected as a chain to the corresponding usb bus. Underneath, it relies on the python [dynamixel sdk](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

**Instantiate**

Each `DynamixelMotorsBus` requires its corresponding usb port (e.g. `DynamixelMotorsBus(port="/dev/tty.usbmodem575E0031751"`). Run our utility script for each arm to find their ports. Here is an example of what it looks like:
```bash
python lerobot/common/robot_devices/motors/dynamixel.py
>>> Finding all available ports for the DynamixelMotorsBus.
>>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
>>> Remove the usb cable from your DynamixelMotorsBus and press Enter when done.
... **Disconnect leader arm and press Enter**
>>> The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751.
>>> Reconnect the usb cable.

python lerobot/common/robot_devices/motors/dynamixel.py
>>> Finding all available ports for the DynamixelMotorsBus.
>>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
>>> Remove the usb cable from your DynamixelMotorsBus and press Enter when done.
... **Disconnect follower arm and press Enter**
>>> The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0032081.
>>> Reconnect the usb cable.
```

Then you can instantiate each arm by listing their motors with their name, motor index, and model. The initial motor index from factory for every motors is `1`. However, unique indices are required for these motors to function in a chain on a common bus. To this end, we set different indices and follow the ascendant convention starting from index `1` (e.g. "1, 2, 3, 4, 5, 6" ). These indices will be written inside the persisting memory of each motor during the first connection. Here is an example of what the instantiation looks like:
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

**Configure and Connect**

During the first connection of the motors, `DynamixelMotorsBus` automatically detects a mismatch between the present motor indices (all `1` by default) and the specified motor indices (e.g. "1, 2, 3, 4, 5, 6"). This triggers the configuration procedure which requires to unplug the power cord and motors, and to sequentially plug each motor again, starting from the closest to the bus. Because it is quite involved, we provide a youtube video for help. The output of the procedure looks like that:
```python
leader_arm.connect()

TODO

follower_arm.connect()

TODO
```

Congrats! Now both arms are well configured and connected. Of course, next time you connect the arms, you won't have to follow configuration procedure ever again. For instance, let's try to disconnect and connect again like that:
```python
leader_arm.disconnect()
leader_arm.connect()

follower_arm.disconnect()
follower_arm.connect()
```

**Read and Write**

Just to get familiar with how `DynamixelMotorsBus` is used to command the motors, let's try to read from them. You should have something like:
```python
values = leader_arm.read("Present_Position")
print(values)
>>> TODO

values = follower_arm.read("Present_Position")
print(values)
>>> TODO
```

The full address is `X_SERIES_CONTROL_TABLE`. TODO

Now let's try to enable torque in the follower arm:
```python
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)

values = follower_arm.read("Present_Position")

values[0] += 10  # Try with positive or negative numbers
follower_arm.write("Goal_Position", values)

follower_arm.write("Goal_Position", values[0], "shoulder_pan")
```


### Teleoperate your Koch v1.1 with KochRobot

**Instantiate**
```python
robot = KochRobot(
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_path=".cache/calibration/koch.pkl",
)
```

**Calibrate and Connect**

```
robot.connect()
>>>
```

```python
degrees = leader_arms.read("Present_Position)
print(degrees)
>>>

degrees = follower_arms.read("Present_Position)
print(degrees)
>>>
```

**Teleoperate**

TODO: explain in pseudo code what the teleop is doing

```python
# Teleoperate for 60 seconds if running at 200 hz
for _ in range(60*200):
    robot.teleop_step()
```

TODO: explain in pseudo code what the teleop(record_data=True) is doing

```python
observation, action = robot.teleop_step(record_data=True)
print(observation)
>>>
print(action)
>>>
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

-
- Improve the dataset
