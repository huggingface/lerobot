# Bring Your Own Hardware

This tutorial will explain how to integrate your own robot design into the LeRobot ecosystem and have it access all of our tools (data collection, control pipelines, policy training and inference).

To that end, we provide the [`Robot`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/robot.py) base class in the LeRobot which specifies a standard interface for physical robot integration. Let's see how to implement it.

## Prerequisites

- Your own robot which exposes a communication interface (e.g. serial, CAN, TCP)
- A way to read sensor data and send motor commands programmatically, e.g. manufacturer's SDK or API, or your own protocol implementation.
- LeRobot installed in your environment. Follow our [Installation Guide](./installation).

## Choose your motors

If you're using Feetech or Dynamixel motors, LeRobot provides built-in bus interfaces:

- [`FeetechMotorsBus`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/feetech.py) – for controlling Feetech servos
- [`DynamixelMotorsBus`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/dynamixel/dynamixel.py) – for controlling Dynamixel servos

Please refer to the [`MotorsBus`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py) abstract class to learn about its API.
For a good example of how it can be used, you can have a look at our own [SO101 follower implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so101_follower/so101_follower.py)

Use these if compatible. Otherwise, you'll need to find or write a Python interface (not covered in this tutorial):

- Find an existing SDK in Python (or use bindings to C/C++)
- Or implement a basic communication wrapper (e.g., via pyserial, socket, or CANopen)

You're not alone—many community contributions use custom boards or firmware!

For Feetech and Dynamixel, we currently support these servos: - Feetech: - STS & SMS series (protocol 0): `sts3215`, `sts3250`, `sm8512bl` - SCS series (protocol 1): `scs0009` - Dynamixel (protocol 2.0 only): `xl330-m077`, `xl330-m288`, `xl430-w250`, `xm430-w350`, `xm540-w270`, `xc430-w150`

If you are using Feetech or Dynamixel servos that are not in this list, you can add those in the [Feetech table](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/feetech/tables.py) or [Dynamixel table](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/dynamixel/tables.py). Depending on the model, this will require you to add model-specific information. In most cases though, there shouldn't be a lot of additions to do.

In the next sections, we'll use a `FeetechMotorsBus` as the motors interface for the examples. Replace it and adapt to your motors if necessary.

## Step 1: Subclass the `Robot` Interface

You’ll first need to specify the config class and a string identifier (`name`) for your robot. If your robot has special needs that you'd like to be able to change easily, it should go here (e.g. port/address, baudrate).

Here, we'll add the port name and one camera by default for our robot:

<!-- prettier-ignore-start -->
```python
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("my_cool_robot")
@dataclass
class MyCoolRobotConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(
        default_factory={
            "cam_1": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=480,
                height=640,
            ),
        }
    )
```
<!-- prettier-ignore-end -->

[Cameras tutorial](./cameras) to understand how to detect and add your camera.

Next, we'll create our actual robot class which inherits from `Robot`. This abstract class defines a contract you must follow for your robot to be usable with the rest of the LeRobot tools.

Here we'll create a simple 5-DoF robot with one camera. It could be a simple arm but notice that the `Robot` abstract class does not assume anything on your robot's form factor. You can let you imagination run wild when designing new robots!

<!-- prettier-ignore-start -->
```python
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots import Robot

class MyCoolRobot(Robot):
    config_class = MyCoolRobotConfig
    name = "my_cool_robot"

    def __init__(self, config: MyCoolRobotConfig):
        super().__init__(config)
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
                "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_4": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_5": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
```
<!-- prettier-ignore-end -->

## Step 2: Define Observation and Action Features

These two properties define the _interface contract_ between your robot and tools that consume it (such as data collection or learning pipelines).

> [!WARNING]
> Note that these properties must be callable even if the robot is not yet connected, so avoid relying on runtime hardware state to define them.

### `observation_features`

This property should return a dictionary describing the structure of sensor outputs from your robot. The keys match what `get_observation()` returns, and the values describe either the shape (for arrays/images) or the type (for simple values).

Example for our 5-DoF arm with one camera:

<!-- prettier-ignore-start -->
```python
@property
def _motors_ft(self) -> dict[str, type]:
    return {
        "joint_1.pos": float,
        "joint_2.pos": float,
        "joint_3.pos": float,
        "joint_4.pos": float,
        "joint_5.pos": float,
    }

@property
def _cameras_ft(self) -> dict[str, tuple]:
    return {
        cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
    }

@property
def observation_features(self) -> dict:
    return {**self._motors_ft, **self._cameras_ft}
```
<!-- prettier-ignore-end -->

In this case, observations consist of a simple dict storing each motor's position and a camera image.

### `action_features`

This property describes the commands your robot expects via `send_action()`. Again, keys must match the expected input format, and values define the shape/type of each command.

Here, we simply use the same joints proprioceptive features (`self._motors_ft`) as with `observation_features`: the action sent will simply the goal position for each motor.

<!-- prettier-ignore-start -->
```python
def action_features(self) -> dict:
    return self._motors_ft
```
<!-- prettier-ignore-end -->

## Step 3: Handle Connection and Disconnection

These methods should handle opening and closing communication with your hardware (e.g. serial ports, CAN interfaces, USB devices, cameras).

### `is_connected`

This property should simply reflect that communication with the robot's hardware is established. When this property is `True`, it should be possible to read and write to the hardware using `get_observation()` and `send_action()`.

<!-- prettier-ignore-start -->
```python
@property
def is_connected(self) -> bool:
    return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
```
<!-- prettier-ignore-end -->

### `connect()`

This method should establish communication with the hardware. Moreover, if your robot needs calibration and is not calibrated, it should start a calibration procedure by default. If your robot needs some specific configuration, this should also be called here.

<!-- prettier-ignore-start -->
```python
def connect(self, calibrate: bool = True) -> None:
    self.bus.connect()
    if not self.is_calibrated and calibrate:
        self.calibrate()

    for cam in self.cameras.values():
        cam.connect()

    self.configure()
```
<!-- prettier-ignore-end -->

### `disconnect()`

This method should gracefully terminate communication with the hardware: free any related resources (threads or processes), close ports, etc.

Here, we already handle this in our `MotorsBus` and `Camera` classes so we just need to call their own `disconnect()` methods:

<!-- prettier-ignore-start -->
```python
def disconnect(self) -> None:
    self.bus.disconnect()
    for cam in self.cameras.values():
        cam.disconnect()
```
<!-- prettier-ignore-end -->

## Step 4: Support Calibration and Configuration

LeRobot supports saving and loading calibration data automatically. This is useful for joint offsets, zero positions, or sensor alignment.

> Note that depending on your hardware, this may not apply. If that's the case, you can simply leave these methods as no-ops:

<!-- prettier-ignore-start -->
```python
@property
def is_calibrated(self) -> bool:
    return True

def calibrate(self) -> None:
    pass
```
<!-- prettier-ignore-end -->

### `is_calibrated`

This should reflect whether your robot has the required calibration loaded.

<!-- prettier-ignore-start -->
```python
@property
def is_calibrated(self) -> bool:
    return self.bus.is_calibrated
```
<!-- prettier-ignore-end -->

### `calibrate()`

The goal of the calibration is twofold:

- Know the physical range of motion of each motors in order to only send commands within this range.
- Normalize raw motors positions to sensible continuous values (e.g. percentages, degrees) instead of arbitrary discrete value dependant on the specific motor used that will not replicate elsewhere.

It should implement the logic for calibration (if relevant) and update the `self.calibration` dictionary. If you are using Feetech or Dynamixel motors, our bus interfaces already include methods to help with this.

<!-- prettier-ignore-start -->
```python
def calibrate(self) -> None:
    self.bus.disable_torque()
    for motor in self.bus.motors:
        self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    input(f"Move {self} to the middle of its range of motion and press ENTER....")
    homing_offsets = self.bus.set_half_turn_homings()

    print(
        "Move all joints sequentially through their entire ranges "
        "of motion.\nRecording positions. Press ENTER to stop..."
    )
    range_mins, range_maxes = self.bus.record_ranges_of_motion()

    self.calibration = {}
    for motor, m in self.bus.motors.items():
        self.calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homing_offsets[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )

    self.bus.write_calibration(self.calibration)
    self._save_calibration()
    print("Calibration saved to", self.calibration_fpath)
```
<!-- prettier-ignore-end -->

### `configure()`

Use this to set up any configuration for your hardware (servos control modes, controller gains, etc.). This should usually be run at connection time and be idempotent.

<!-- prettier-ignore-start -->
```python
def configure(self) -> None:
    with self.bus.torque_disabled():
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", motor, 16)
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 32)
```
<!-- prettier-ignore-end -->

## Step 5: Implement Sensors Reading and Action Sending

These are the most important runtime functions: the core I/O loop.

### `get_observation()`

Returns a dictionary of sensor values from the robot. These typically include motor states, camera frames, various sensors, etc. In the LeRobot framework, these observations are what will be fed to a policy in order to predict the actions to take. The dictionary keys and structure must match `observation_features`.

<!-- prettier-ignore-start -->
```python
def get_observation(self) -> dict[str, Any]:
    if not self.is_connected:
        raise ConnectionError(f"{self} is not connected.")

    # Read arm position
    obs_dict = self.bus.sync_read("Present_Position")
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

    # Capture images from cameras
    for cam_key, cam in self.cameras.items():
        obs_dict[cam_key] = cam.async_read()

    return obs_dict
```
<!-- prettier-ignore-end -->

### `send_action()`

Takes a dictionary that matches `action_features`, and sends it to your hardware. You can add safety limits (clipping, smoothing) and return what was actually sent.

For simplicity, we won't be adding any modification of the actions in our example here.

<!-- prettier-ignore-start -->
```python
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

    # Send goal position to the arm
    self.bus.sync_write("Goal_Position", goal_pos)

    return action
```
<!-- prettier-ignore-end -->

## Adding a Teleoperator

For implementing teleoperation devices, we also provide a [`Teleoperator`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/teleoperators/teleoperator.py) base class. This class is very similar to the `Robot` base class and also doesn't assume anything on form factor.

The main differences are in the I/O functions: a teleoperator allows you to produce action via `get_action` and can receive feedback actions via `send_feedback`. Feedback could be anything controllable on the teleoperation device that could help the person controlling it understand the consequences of the actions sent. Think motion/force feedback on a leader arm, vibrations on a gamepad controller for example. To implement a teleoperator, you can follow this same tutorial and adapt it for these two methods.

## Using Your Own `LeRobot` Devices 🔌

You can easily extend `lerobot` with your own custom hardware—be it a camera, robot, or teleoperation device—by creating a separate, installable Python package. If you follow a few simple conventions, the `lerobot` command-line tools (like `lerobot-teleop` and `lerobot-record`) will **automatically discover and integrate your creations** without requiring any changes to the `lerobot` source code.

This guide outlines the conventions your plugin must follow.

### The 4 Core Conventions

To ensure your custom device is discoverable, you must adhere to the following four rules.

#### 1\. Create an Installable Package with a Specific Prefix

Your project must be a standard, installable Python package. Crucially, the name of your package (as defined in `pyproject.toml` or `setup.py`) must begin with one of these prefixes:

- `lerobot_robot_` for a robot.
- `lerobot_camera_` for a camera.
- `lerobot_teleoperator_` for a teleoperation device.

This prefix system is how `lerobot` automatically finds your plugin in the Python environment.

#### 2\. Follow the `SomethingConfig`/`Something` Naming Pattern

Your device's implementation class must be named after its configuration class, simply by removing the `Config` suffix.

- **Config Class:** `MyAwesomeTeleopConfig`
- **Device Class:** `MyAwesomeTeleop`

#### 3\. Place Your Files in a Predictable Structure

The device class (`MyAwesomeTeleop`) must be located in a predictable module relative to its configuration class (`MyAwesomeTeleopConfig`). `lerobot` will automatically search in these locations:

- In the **same module** as the config class.
- In a **submodule named after the device** (e.g., `my_awesome_teleop.py`).

The recommended and simplest structure is to place them in separate, clearly named files within the same directory.

#### 4\. Expose Classes in `__init__.py`

Your package's `__init__.py` file should import and expose both the configuration and the device classes, making them easily accessible.

### Putting It All Together: A Complete Example

Let's create a new teleoperator called `my_awesome_teleop`.

#### Directory Structure

Here is what the project folder should look like. The package name, `lerobot_teleoperator_my_awesome_teleop`, follows **Convention \#1**.

```
lerobot_teleoperator_my_awesome_teleop/
├── pyproject.toml # (or setup.py) lists lerobot as a dependency
└── lerobot_teleoperator_my_awesome_teleop/
    ├── __init__.py
    ├── config_my_awesome_teleop.py
    └── my_awesome_teleop.py
```

#### File Contents

- **`config_my_awesome_teleop.py`**: Defines the configuration class. Note the `Config` suffix (**Convention \#2**).

  ```python
  from dataclasses import dataclass

  from lerobot.teleoperators.config import TeleoperatorConfig

  @TeleoperatorConfig.register_subclass("my_awesome_teleop")
  @dataclass
  class MyAwesomeTeleopConfig(TeleoperatorConfig):
      # Your configuration fields go here
      port: str = "192.168.1.1"
  ```

- **`my_awesome_teleop.py`**: Implements the device. The class name `MyAwesomeTeleop` matches its config class name (**Convention \#2**). This file structure adheres to **Convention \#3**.

  ```python
  from lerobot.teleoperators.teleoperator import Teleoperator

  from .config_my_awesome_teleop import MyAwesomeTeleopConfig

  class MyAwesomeTeleop(Teleoperator):
      config_class = MyAwesomeTeleopConfig
      name = "my_awesome_teleop"

      def __init__(self, config: MyAwesomeTeleopConfig):
          super().__init__(config)
          self.config = config

      # Your device logic (e.g., connect) goes here
  ```

- **`__init__.py`**: Exposes the key classes (**Convention \#4**).

  ```python
  from .config_my_awesome_teleop import MyAwesomeTeleopConfig
  from .my_awesome_teleop import MyAwesomeTeleop
  ```

### Installation and Usage

1.  **Install your new plugin in your Python environment.** You can install your local plugin package using `pip`'s editable mode or from PyPi.

    ```bash
    # Locally
    # Navigate to your plugin's root directory and install it
    cd lerobot_teleoperator_my_awesome_teleop
    pip install -e .

    # From PyPi
    pip install lerobot_teleoperator_my_awesome_teleop
    ```

2.  **Use it directly from the command line.** Now, you can use your custom device by referencing its type.

    ```bash
    lerobot-teleoperate --teleop.type=my_awesome_teleop \
    # other arguments
    ```

And that's it\! Your custom device is now fully integrated.

### Looking for an example ?

Check out these two packages from the community:

- https://github.com/SpesRobotics/lerobot-robot-xarm
- https://github.com/SpesRobotics/lerobot-teleoperator-teleop

## Wrapping Up

Once your robot class is complete, you can leverage the LeRobot ecosystem:

- Control your robot with available teleoperators or integrate directly your teleoperating device
- Record training data and visualize it
- Integrate it into RL or imitation learning pipelines

Don't hesitate to reach out to the community for help on our [Discord](https://discord.gg/s3KuuzsPFb) 🤗
