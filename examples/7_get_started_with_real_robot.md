# Robots in the real-world

This tutorial explains how to get started with real robots and train a neural network to control them autonomously.

It covers how to:
1. order and assemble your robot,
2. connect your robot, configure and calibrate it,
3. record your dataset and visualize it,
4. train a policy on your data and make sure it's ready for evaluation,
5. evaluate your policy and visualize the result afterwards.

Following these steps, you will reproduce behaviors like picking a lego block and placing it in a bin with a relatively high success rate like in this video: https://x.com/RemiCadene/status/1814680760592572934

While this tutorial is general and easily extendable to any type of robots by changing a configuration, it is based on the [Koch v1.1](https://github.com/jess-moss/koch-v1-1) affordable robot. Koch v1.1 is composed of a leader arm and a follower arm with 6 motors each. In addition, various cameras can record the scene and serve as visual sensors for the robot.

During data collection, you will control the follower arm by moving the leader arm. This is called "teleoperation". It is used to collect robot trajectories.
Then, you will train a neural network to imitate these trajectories. Finally, you will deploy your neural network to autonomously control your robot.

Note: If you have issue at any step of the tutorial, ask for help on [Discord](https://discord.com/invite/s3KuuzsPFb).

## 1. Order and Assemble your Koch v1.1

Follow the sourcing and assembling instructions on the [Koch v1.1 github page](https://github.com/jess-moss/koch-v1-1) to setup the follower and leader arms shown in this picture.

<div style="text-align:center;">
  <img src="../media/koch/leader_follower.webp?raw=true" alt="Koch v1.1 leader and follower arms" title="Koch v1.1 leader and follower arms" width="50%">
</div>

## 2. Connect, Configure, and Calibrate your Koch v1.1

Connect the leader arm (the smaller one) with the 5V alimentation and the follower arm with the 12V alimentation. Then connect both arms to your computer with USB.

### Control your motors with DynamixelMotorsBus

You can use the [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py) to efficiently read from and write to the motors connected as a chain to the corresponding usb bus. Underneath, it relies on the python [dynamixel sdk](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

**Instantiate**

You will need to instantiate two [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py), one for each arm, with their corresponding usb port (e.g. `DynamixelMotorsBus(port="/dev/tty.usbmodem575E0031751"`).

To find their corresponding ports, run our utility script twice:
```bash
python lerobot/common/robot_devices/motors/dynamixel.py
```

A first time to find the port of the leader arm (e.g. `/dev/tty.usbmodem575E0031751`):
```
Finding all available ports for the DynamixelMotorsBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your DynamixelMotorsBus and press Enter when done.

**Disconnect leader arm and press Enter**

The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751
Reconnect the usb cable.
```

A second time to find the port of the follower arm (e.g. `/dev/tty.usbmodem575E0032081`):
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

Update the corresponding ports of this code with your ports and run the code to instantiate the Koch leader and follower arms:
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

Also, update the ports of the following lines of yaml file for Koch robot [`lerobot/configs/robot/koch.yaml`](../lerobot/configs/robot/koch.yaml):
```yaml
[...]
leader_arms:
  main:
    _target_: lerobot.common.robot_devices.motors.dynamixel.DynamixelMotorsBus
    port: /dev/tty.usbmodem575E0031751  # <- Update
    motors:
      # name: (index, model)
      shoulder_pan: [1, "xl330-m077"]
      shoulder_lift: [2, "xl330-m077"]
      elbow_flex: [3, "xl330-m077"]
      wrist_flex: [4, "xl330-m077"]
      wrist_roll: [5, "xl330-m077"]
      gripper: [6, "xl330-m077"]
follower_arms:
  main:
    _target_: lerobot.common.robot_devices.motors.dynamixel.DynamixelMotorsBus
    port: /dev/tty.usbmodem575E0032081  # <- Update
    motors:
      # name: (index, model)
      shoulder_pan: [1, "xl430-w250"]
      shoulder_lift: [2, "xl430-w250"]
      elbow_flex: [3, "xl330-m288"]
      wrist_flex: [4, "xl330-m288"]
      wrist_roll: [5, "xl330-m288"]
      gripper: [6, "xl330-m288"]
[...]
```

This file is used to instantiate your robot in all our scripts. We will explain how this works later on.

**Configure and Connect**

Then, you will need to configure your motors to be able to properly communicate with them. During the first connection of the motors, [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py) automatically detects a mismatch between the present motor indices (all `1` by factory default) and your specified motor indices (e.g. `1,2,3,4,5,6`). This triggers the configuration procedure which requires to unplug the power cord and motors, and to sequentially plug each motor again, starting from the closest to the bus.

Take a look at this youtube video for help: TODO(rcadene)

Run the following code in the same python session in your terminal to connect and configure the leader arm:
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

*Follow the procedure*

Setting expected motor indices: [1, 2, 3, 4, 5, 6]
```

Now do the same for the follower arm:
```python
follower_arm.connect()
```

Congrats, now both arms are well configured and connected! You won't have to follow the configuration procedure ever again!

Note: If the configuration didn't work, you might need to update the firmware using [DynamixelWizzard2](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2). You might also need to manually configure the motors. Similarly, you will need to connect each motor seperately to the bus. You will need to set correct indices and set their baudrates to `1000000`. Take a look at this youtube video for help: https://www.youtube.com/watch?v=JRRZW_l1V-U


**Read and Write**

Just to get familiar with how `DynamixelMotorsBus` communicates with the motors, let's try to read from them:
```python
leader_pos = leader_arm.read("Present_Position")
follower_pos = follower_arm.read("Present_Position")
print(leader_pos)
print(follower_pos)
>>> array([2054,  523, 3071, 1831, 3049, 2441], dtype=int32)
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

Before being able to teleoperate your robot, you will need to instantiate the [`KochRobot`](../lerobot/common/robot_devices/robots/koch.py) using the previously defined `leader_arm` and `follower_arm` as shown next.

For the Koch robot, we only have one leader, so as you will see next, we call it `"main"` and define `leader_arms={"main": leader_arm}`. We do the same for the follower arm. However for other robots (e.g. Aloha), we can use two pairs of leader and follower. In this case, we would define `leader_arms={"left": left_leader_arm, "right": right_leader_arm},`. Same thing for the follower arms.

We also need to provide a path to a calibration file `calibration_path=".cache/calibration/koch.pkl"`. More on this in the next section.

Run this code to instantiate your robot:
```python
from lerobot.common.robot_devices.robots.koch import KochRobot

robot = KochRobot(
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_path=".cache/calibration/koch.pkl",
)
```

**Calibrate and Connect**

Then, you will need to calibrate your robot so that when the leader and follower arms are in the same physical position, they have the same position values read from [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py). An important benefit of calibration is that a neural network trained on data collected on your Koch robot will transfer to another Koch robot.

During the first connection of your robot, [`KochRobot`](../lerobot/common/robot_devices/robots/koch.py) detects that the calibration file is missing. This triggers the calibration procedure which requires you to move each arm in 3 different positions.

You will follow the procedure and move the follower to these positions:

<div style="text-align:center; display:flex; justify-content:center;">
  <figure style="margin: 10px;">
    <img src="../media/koch/follower_zero.webp?raw=true" alt="Koch v1.1 follower arm zero position" title="Koch v1.1 follower arm zero position" width="100%">
    <figcaption>1. Zero position</figcaption>
  </figure>
  <figure style="margin: 10px;">
    <img src="../media/koch/follower_rotated.webp?raw=true" alt="Koch v1.1 follower arm rotated position" title="Koch v1.1 follower arm rotated position" width="100%">
    <figcaption>2. Rotated position</figcaption>
  </figure>
  <figure style="margin: 10px;">
    <img src="../media/koch/follower_rest.webp?raw=true" alt="Koch v1.1 follower arm rest position" title="Koch v1.1 follower arm rest position" width="100%">
    <figcaption>3. Rest position</figcaption>
  </figure>
</div>

Then you will continue the procedure and move the leader to these positions:

<div style="text-align:center; display:flex; justify-content:center;">
  <figure style="margin: 10px;">
    <img src="../media/koch/leader_zero.webp?raw=true" alt="Koch v1.1 leader arm zero position" title="Koch v1.1 leader arm zero position" width="100%">
    <figcaption>1. Zero position</figcaption>
  </figure>
  <figure style="margin: 10px;">
    <img src="../media/koch/leader_rotated.webp?raw=true" alt="Koch v1.1 leader arm rotated position" title="Koch v1.1 leader arm rotated position" width="100%">
    <figcaption>2. Rotated position</figcaption>
  </figure>
  <figure style="margin: 10px;">
    <img src="../media/koch/leader_rest.webp?raw=true" alt="Koch v1.1 leader arm rest position" title="Koch v1.1 leader arm rest position" width="100%">
    <figcaption>3. Rest position</figcaption>
  </figure>
</div>

Run this code to calibrate and connect your robot:
```python
robot.connect()

>>> TODO
```

Now we will see how to read the position of the leader and follower arms using `read` from [`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py). If the calibration is done well, the posiiton should be the similar when both arms are in a similar position.

Run this code to get the positions in degree:
```python
leader_pos = robot.leader_arms["main"].read("Present_Position")
print(leader_pos)
follower_pos = robot.follower_arms["main"].read("Present_Position")
print(follower_pos)
>>> array([-0.43945312, 133.94531, 179.82422, -18.984375, -1.9335938, 34.541016], dtype=float32)
>>> array([-1.0546875, 128.67188, 174.90234, -7.998047, -5.4492188, 32.34375], dtype=float32)
```

Importantly, we also converted the "step" position to degree. This is much easier to interpet and debug. In particular, the zero position used during calibration now corresponds to 0 degree for each motor. Also, the rotated position corresponds to 90 degree for each motor.

**Teleoperate**

Now you can easily teleoperate your robot by reading the positions from the leader arm and sending them as goal positions to the follower arm.

Run this code to teleoperate:
```python
import tqdm
# Teleoperate for 30 seconds
# Fastest communication is done at a frequency of ~200Hz
for _ in tqdm.tqdm(range(30*200)):
    leader_pos = robot.leader_arms["main"].read("Present_Position")
    robot.follower_arms["main"].write("Goal_Position", leader_pos)
```

You can also teleoperate by using `teleop_step` from [`KochRobot`](../lerobot/common/robot_devices/robots/koch.py).

Run this code to teleoperate:
```python
import tqdm
for _ in tqdm.tqdm(range(30*200)):
    robot.teleop_step()
```

Teleoperation is useful to record data. To this end, you can use `teleop_step` from [`KochRobot`](../lerobot/common/robot_devices/robots/koch.py) with `record_data=True`. It outputs the follower position as `"observation.state"` and the leader position as `"action"`. It also converts the numpy arrays into torch tensors, and concatenates the positions in the case of two leader and two follower arms like in Aloha.

Run this code to try data recording during a teleoperation step:
```python
observation, action = robot.teleop_step(record_data=True)
leader_pos = robot.leader_arms["main"].read("Present_Position")
follower_pos = robot.follower_arms["main"].read("Present_Position")
print(follower_pos)
TODO
print(observation)
print(leader_pos)
print(action)
>>> {'observation.state': tensor([7.8223, 131.1328, 165.5859, -23.4668, -0.9668, 32.4316])}
>>> {'action': tensor([3.4277, 134.1211, 179.8242, -18.5449, -1.5820, 34.7168])}
```

Finally, `teleop_step` from [`KochRobot`](../lerobot/common/robot_devices/robots/koch.py) with `record_data=True` can also asynchrously record frames from several cameras and add them to the observation dictionnary as `"observation.images.CAMERA_NAME"`. More on this in the next section.

When you are done, disconnect your robot:
```python
robot.disconnect()
```

### Add your cameras with OpenCVCamera

**Instantiate**

You can efficiently record frames from cameras with the [`OpenCVCamera`](../lerobot/common/robot_devices/cameras/opencv.py) class. It relies on [`opencv2`](https://docs.opencv.org) to communicate with the cameras. Most cameras are compatible. For more info, see [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

To instantiate an [`OpenCVCamera`](../lerobot/common/robot_devices/cameras/opencv.py), you need a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera like a webcam of a laptop, the camera index is usually `0` but it might differ, and the camera index might change if you reboot your computer or re-plug your camera. This behavior depends on your operating system.

To find the camera indices of your cameras, you can run our utility script that will save a few frames for each camera:
```bash
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```

The output looks like this for two cameras:
```
Mac or Windows detected. Finding available camera indices through scanning all indices from 0 to 60
[...]
Camera found at index 0
Camera found at index 1
[...]
Connecting cameras
OpenCVCamera(0, fps=30.0, width=1920.0, height=1080.0, color_mode=rgb)
OpenCVCamera(1, fps=24.0, width=1920.0, height=1080.0, color_mode=rgb)
Saving images to outputs/images_from_opencv_cameras
Frame: 0000	Latency (ms): 39.52
[...]
Frame: 0046	Latency (ms): 40.07
Images have been saved to outputs/images_from_opencv_cameras
```

Then, look at the saved images in `outputs/images_from_opencv_cameras` to know which camera index (e.g. `0` for `camera_00` or `1` for `camera_01`) is associated to which physical camera:
```
camera_00_frame_000000.png
[...]
camera_00_frame_000047.png
camera_01_frame_000000.png
[...]
camera_01_frame_000047.png
```

Note: We save a few frames since some cameras need a few seconds to warmup. The first frame can be totally black.

Finally, run this code to instantiate your camera:
```python
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

camera = OpenCVCamera(camera_index=0)
camera.connect()
color_image = camera.read()
print(color_image.shape)
print(color_image.dtype)
>>> (1080, 1920, 3)
>>> uint8
```

Note that default fps, width, height and color_mode of the given camera are used. They may differ for different cameras. You can specify them during instantiation (e.g. `OpenCVCamera(camera_index=0, fps=30, width=640, height=480`).

When done using the camera, disconnect it:
```python
camera.disconnect()
```

**Instantiate your robot with cameras**

You can also instantiate your robot with your cameras!

Run this code:
```python
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
```

As a result, `teleop_step` with `record_data=True` will return a frame for each camera following the torch convention of channel first.

Run this code:
```python
observation, action = robot.teleop_step(record_data=True)
print(observation)
print(action)
>>> TODO
>>> TODO
```

Also, update the flollowing lines of the yaml file for Koch robot [`lerobot/configs/robot/koch.yaml`](../lerobot/configs/robot/koch.yaml) with your cameras and their corresponding camera indices:
```yaml
[...]
cameras:
  laptop:
    _target_: lerobot.common.robot_devices.cameras.opencv.OpenCVCamera
    camera_index: 0
    fps: 30
    width: 640
    height: 480
  phone:
    _target_: lerobot.common.robot_devices.cameras.opencv.OpenCVCamera
    camera_index: 1
    fps: 30
    width: 640
    height: 480
```

This file is used to instantiate your robot in all our scripts. We will explain how this works in the next section.

### Use `koch.yaml` and our `teleoperate` function

Instead of manually running the python code in a terminal window, you can use [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) to instantiate your robot by providing the path to the robot yaml file (e.g. [`lerobot/configs/robot/koch.yaml`](../lerobot/configs/robot/koch.yaml) and control your robot with various modes as explained next.

Run this code to teleoperate your robot:
```bash
python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml

>>> TODO
```

Note, you can override any entry in the yaml file using `--robot-overrides` and the [Hydra synthax](https://hydra.cc/docs). If needed, you can override the ports like this:
```bash
python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml \
  --robot-overrides \
    leader_arms.main.port=/dev/tty.usbmodem575E0031751 \
    follower_arms.main.port=/dev/tty.usbmodem575E0032081
```

If you don't have any camera, you can remove them dynamically with this weird hydra synthax `'~cameras'`:
```bash
python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml \
  --robot-overrides \
    leader_arms.main.port=/dev/tty.usbmodem575E0031751 \
    follower_arms.main.port=/dev/tty.usbmodem575E0032081
    '~cameras'
```

We advise to create a new yaml file when the command becomes too long.

## 3. Record your Dataset and Visualize it

Using what you've learned previously, you can now easily record a dataset of states and actions for one episode. You can use `busy_wait` to control the speed of teleoperation and record at a fixed `fps` (frame per seconds).

Try this code to record 30 seconds at 60 fps:
```python
from lerobot.scripts.control_robot import busy_wait

record_time_s = 30
fps = 60

states = []
actions = []
for _ in range(record_time_s * fps):
    start_time = time.perf_counter()
    observation, action = robot.teleop_step(record_data=True)

    states.append(observation["observation.state"])
    actions.append(action["action"])

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)
```

However, many utilities are still missing. For instance, if you have cameras, you will need to save the images on disk to not go out of RAM, and to do so in threads to not slow down communication with your robot. Also, you will need to store your data in a format optimized for training and web sharing like [`LeRobotDataset`](../lerobot/common/datasets/lerobot_dataset.py). More on this in the next section.

### Use `koch.yaml` and the `record` function

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) to achieve efficient data recording. It encompasses many recording utilities:
1. Frames from cameras are saved on disk in threads, and encoded into videos at the end of recording.
2. Video streams from cameras are displayed in window so that you can verify them.
3. Data is stored with [`LeRobotDataset`](../lerobot/common/datasets/lerobot_dataset.py) format which is pushed to your Hugging Face page.
4. Checkpoints are done during recording, so if any issue occurs, you can resume recording by re-running the same command again. You can also use `--force-override 1` to start recording from scratch.
5. Set the flow of data recording using command line arguments:
  - `--warmup-time-s` defines the number of seconds before starting data collection. It allows the robot devices to warmup and synchronize (10 seconds by default).
  - `--episode-time-s` defines the number of seconds for data recording for each episode (60 seconds by default).
  - `--reset-time-s` defines the number of seconds for resetting the environment after each episode (60 seconds by default).
  - `--num-episodes` defines the number of episodes to record (50 by default).
6. Control the flow during data recording using keyboard keys:
  - Press right arrow `->` at any time during episode recording to early stop and go to resetting. Same during resetting, to early stop and to go to the next episode recording.
  - Press left arrow `<-` at any time during episode recording or resetting to early stop, cancel the current episode, and re-record it.
  - Press escape `ESC` at any time during episode recording to end the session early and go straight to video encoding and dataset uploading.
7. Similarly to `teleoperate`, you can also use `--robot-path` and `--robot-overrides` to specify your robots.

Before trying `record`, make sure you've logged in using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

And store your Hugging Face repositery name in a variable (e.g. `cadene` or `lerobot`). For instance, run this to use your Hugging Face user name as repositery:
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Now run this to record 5 episodes:
```bash
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_test \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 5
```

It will output something like:
```
TODO
```

At the end, your dataset will be uploaded on your Hugging Face page (e.g. https://huggingface.co/datasets/cadene/koch_test) that you can obtain by running:
```bash
echo https://huggingface.co/datasets/${HF_USER}/koch_test
```

### Advices for recording dataset

Now that you are used to data recording, you can record a bigger dataset for training. A good hello world task consists in grasping an object at various locations and placing it in a bin. We recommend to record a minimum of 50 episodes with 10 episodes per location, to not move the cameras and to grasp with a consistent behavior.

In the next sections, you will train your neural network. Once it can grasp pretty well, you can introduce slightly more variance in during data collection such as more grasp locations, various grasping behaviors, various positions for the cameras, etc.

### Visualize all episodes

You can visualize your dataset by running:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --root data \
  --repo-id ${HF_USER}/koch_test
```

This will launch a local web server that looks like this:
<div style="text-align:center;">
  <img src="../media/tutorial/visualize_dataset_html.webp?raw=true" alt="Koch v1.1 leader and follower arms" title="Koch v1.1 leader and follower arms" width="100%">
</div>

### Replay episode on your robot with the `replay` function

Another cool function of [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) is `replay` which allows to replay on your robot any episode that you've recorded or from any dataset out there. It's a way to test repeatability of your robot and transferability across robots of the same type.

Run this to replay the first episode of the dataset you've just recorded:
```bash
python lerobot/scripts/control_robot.py replay \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_test \
  --episode 0
```

Your robot should reproduce very similar movements as what you recorded. For instance, see this video where we use `replay` on a Aloha robot from [Trossen Robotics](https://www.trossenrobotics.com): https://x.com/RemiCadene/status/1793654950905680090

## 4. Train a policy on your data

### Use our `train` script

Then, you can train a policy to control your robot using the [`lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required.

Firstly, provide your dataset with `dataset_repo_id=${HF_USER}/koch_test`.

Secondly, provide a policy with `policy=act_koch_real`. This loads configurations from [`lerobot/configs/policy/act_koch_real.yaml`](../lerobot/configs/policy/act_koch_real.yaml). Importantly, this policy uses 2 cameras as input `laptop` and `phone`. If your dataset has different cameras, update the yaml file to account for it in the following parts:
```yaml
...
override_dataset_stats:
  observation.images.laptop:
    # stats from imagenet, since we use a pretrained vision model
    mean: [[[0.485]], [[0.456]], [[0.406]]]  # (c,1,1)
    std: [[[0.229]], [[0.224]], [[0.225]]]  # (c,1,1)
  observation.images.phone:
    # stats from imagenet, since we use a pretrained vision model
    mean: [[[0.485]], [[0.456]], [[0.406]]]  # (c,1,1)
    std: [[[0.229]], [[0.224]], [[0.225]]]  # (c,1,1)
...
  input_shapes:
    observation.images.laptop: [3, 480, 640]
    observation.images.phone: [3, 480, 640]
...
  input_normalization_modes:
    observation.images.laptop: mean_std
    observation.images.phone: mean_std
...
```

Thirdly, provide an environment with `env=koch_real`. This loads configurations from [`lerobot/configs/env/koch_real.yaml`](../lerobot/configs/env/koch_real.yaml). It looks like
```yaml
fps: 30
env:
  name: real_world
  task: null
  state_dim: 6
  action_dim: 6
  fps: ${fps}
```
It should match your dataset (e.g. `fps: 30`) and your robot (e.g. `state_dim: 6` and `action_dim: 6`). We are still working on simplifying this in future versions of `lerobot`.

Optionnaly, you can use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots with `wandb.enable=true`. Make sure you are logged in by running `wandb login`.

Finally, use `DATA_DIR=data` to access your dataset stored in your local `data` directory. If you dont provide `DATA_DIR`, your dataset will be downloaded from Hugging Face hub to your cache folder `$HOME/.cache/hugginface`. In future versions of `lerobot`, both directories will be in sync.

Now, start training:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/koch_test \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/act_koch_test \
  hydra.job.name=act_koch_test \
  wandb.enable=true
```

For more information on the `train` script see the previous tutorial: [`examples/4_train_policy_with_script.md`](../examples/4_train_policy_with_script.md)

## Upload policy checkpoints to the hub

Once training is done, upload the latest checkpoint with:
```bash
huggingface-cli upload ${HF_USER}/act_koch_test \
  outputs/train/act_koch_test/checkpoints/last/pretrained_model
```

You can also upload intermediate checkpoints with:
```bash
CKPT=010000
huggingface-cli upload ${HF_USER}/act_koch_test_${CKPT} \
  outputs/train/act_koch_test/checkpoints/${CKPT}/pretrained_model
```

### Visualize predictions on training set

Optionnaly, you can visualize the predictions of your neural network on your training data. This is a useful debugging tool. You can provide a checkpoint directory as input (e.g. `outputs/train/act_koch_test/checkpoints/last/pretrained_model`). For instance:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/koch_test \
  --episodes 0 1 2 \
  -p outputs/train/act_koch_test/checkpoints/last/pretrained_model
```

You can also provide a model repository as input (e.g. `${HF_USER}/act_koch_test`). For instance:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/koch_test \
  --episodes 0 1 2 \
  -p ${HF_USER}/act_koch_test
```

## 5. Evaluate your policy

Now that you have a policy checkpoint, you can easily control your robot with it using:
- `observation = robot.capture_observation()`
- `action = policy.select_action(observation)`
- `robot.send_action(action)`

Try this code for running inference for 60 seconds at 30 fps:
```python
from lerobot.common.policies.act.modeling_act import ActPolicy

inference_time_s = 60
fps = 30

ckpt_path = "outputs/train/act_koch_test/checkpoints/last/pretrained_model"
policy = ActPolicy.from_pretrained(ckpt_path)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].cuda()

    action = policy.select_action(observation)
    robot.send_action(action)

    # remove batch dimension
    action = action.squeeze(0)
    action = action.to("cpu")

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)
```

### Use `koch.yaml` and our `record` function

Ideally, when controlling your robot with your neural network, you would want to record evaluation episodes and to be able to visualize them later on, or even train on them like in Reinforcement Learning. This pretty much corresponds to recording a new dataset but with a neural network providing the actions instead of teleoperation.

To this end, you can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. Just copy the same command as previously used to record your training dataset and change two things:
1. Add a path to your policy checkpoint with `-p` (e.g. `-p outputs/train/eval_koch_test/checkpoints/last/pretrained_model`) or a model repository (e.g. `-p ${HF_USER}/act_koch_test`).
2. Change the dataset name to reflect you are running inference (e.g. `--repo-id ${HF_USER}/eval_koch_test`).

Now run this to record 5 evaluation episodes.
```bash
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/eval_koch_test \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 5 \
  -p outputs/train/act_koch_test/checkpoints/last/pretrained_model
```

### Visualize evaluation afterwards

You can then visualize your evaluation dataset by running the same command as before but with the new dataset as argument:
```bash
python lerobot/scripts/visualize_dataset.py \
  --root data \
  --repo-id ${HF_USER}/eval_koch_test
```
