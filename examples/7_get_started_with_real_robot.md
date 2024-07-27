# Robots in the real-world

This tutorial explains how to get started with real robots and train a neural network to control them autonomously.

It covers how to:
- order and assemble your robot,
- connect your robot, configure it and calibrate it,
- record your dataset and visualize it,
- train a policy on your data and make sure it's ready for evaluation,
- evaluate your policy and visualize the result afterwards.

Following these steps, you should be able to reproduce behaviors like picking a lego block and placing it in a bin with a relatively high success rate.

While this tutorial is general and easily extendable to any type of robots by changing a configuration, it is based on the [Koch v1.1](https://github.com/jess-moss/koch-v1-1) affordable robot. Koch v1.1 is composed of a leader arm and a follower arm with 6 motors each. In addition, various cameras can record the scene and serve as visual sensors for the robot.

During data collection, you will control the follower arm by moving the leader arm. This is called "teleoperation". More specifically, the present position of the motors of the leader arm is read at high frequency and sent as a goal position for the motors of the follower arm, which effectively "follow" the movements of the leader arm. While you teleoperate the robot, a few modalities are recorded:
- the present position of the follower arm, called the "state",
- the goal position sent to the follower arm, called the "action",
- the video stream from the cameras.

Finally, you will train a neural network to predict the future actions given the state and camera frames as input ; and deploy it to autonomously control the robot via the high frequency communication of goal positions to the follower arm.


## 1. Order and Assemble Koch v1.1

Follow the bill of materials on the [Koch v1.1 github page](https://github.com/jess-moss/koch-v1-1) to order a leader and a follower arm. Some parts and prices are a bit different with respect to the geo location.

Once the parts are received, follow this video to guide you through the assembly:

## 2. Connect, configure, and calibrate Koch v1.1

Connect the leader arm (the smaller one) with the 5V alimentation and the follower arm with the 12V alimentation. Then connect both arms to your computer with USB.

### DynamixelMotorsBus

[`DynamixelMotorsBus`](lerobot/common/robot_devices/motors/dynamixel.py) has been made to communicate with each arm. This class allows to efficiently read from and write to the motors plugged to the corresponding usb bus. Underneath, it relies on the python [dynamixel sdk](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

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

Then you can instantiate each arm by listing their motors with their name, motor index, and model. The initial motor index from factory for every motors is `1`. However, unique indices are required for these motors to function in a chain on a common bus. To this end, we set different indices and follow the ascendant convention starting from index `1`. These indices will be written inside the persisting memory of each motor during the first connection. Here is an example of what the instantiation looks like:
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

During the first connection of the motors, `DynamixelMotorsBus` automatically detects a mismatch between the present motor indices (all `1` by default) and the specified motor indices. This triggers the configuration procedure which requires to unplug the power cord and motors, and to sequentially plug each motor again, starting from the closest to the bus. Because it is quite involved, we provide a youtube video for help. The output of the procedure looks like that:
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


### KochRobot

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

```python
# Teleoperate for 60 seconds if running at 200 hz
for _ in range(60*200):
    robot.teleop_step()
```

```python
robot.teleop_step(record_data=True)
```
