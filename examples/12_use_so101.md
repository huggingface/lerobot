# Assemble and use SO-101

In the steps below we explain how to assemble and use our flagship robot, the SO-101 with LeRobot 🤗.

## Source the parts

Follow this [README](https://github.com/TheRobotStudio/SO-ARM100). It contains the bill of materials, with a link to source the parts, as well as the instructions to 3D print the parts,
and advice if it's your first time printing or if you don't own a 3D printer.

Before assembling, you will first need to configure your motors. To this end, we provide a nice script, so let's first install LeRobot. After configuration, we will also guide you through assembly.

## Install LeRobot

> [!TIP]
> We use the Command Prompt (cmd) quite a lot. If you are not comfortable using the cmd or want to brush up using the command line you can have a look here: [Command line crash course](https://developer.mozilla.org/en-US/docs/Learn_web_development/Getting_started/Environment_setup/Command_line)

Download our source code:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/miniconda/install/#quick-command-line-install):
```bash
conda create -y -n lerobot python=3.10
```
Now restart the shell by running:

##### Windows:
```bash
`source ~/.bashrc`
```

##### Mac:
```bash
`source ~/.bash_profile`
```

##### zshell:
```bash
`source ~/.zshrc`
```

Then activate your conda environment, you have to do this each time you open a shell to use lerobot:
```bash
conda activate lerobot
```

When using `miniconda`, install `ffmpeg` in your environment:
```bash
conda install ffmpeg -c conda-forge
```

> [!NOTE]
> This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>  - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>  ```bash
>  conda install ffmpeg=7.1.1 -c conda-forge
>  ```
>  - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

Install 🤗 LeRobot:
```bash
cd lerobot && pip install -e ".[feetech]"
```

> [!NOTE]
> If you encounter build errors, you may need to install additional dependencies (`cmake`, `build-essential`, and `ffmpeg libs`). On Linux, run: `sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config`. For other systems, see: [Compiling PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)


## Configure motors

To configure the motors designate one bus servo adapter and 6 motors for your leader arm, and similarly the other bus servo adapter and 6 motors for the follower arm. It's convenient to label them and write on each motor if it's for the follower `F` or for the leader `L` and it's ID from 1 to 6.

You now should plug the 5V or 12V power supply to the motor bus. 5V for the STS3215 7.4V motors and 12V for the STS3215 12V motors. Note that the leader arm always uses the 7.4V motors, so watch out that you plug in the right power supply if you have 12V and 7.4V motors, otherwise you might burn your motors! Now, connect the motor bus to your computer via USB. Note that the USB doesn't provide any power, and both the power supply and USB have to be plugged in.

### Find the USB ports associated to each arm

To find the port for each bus servo adapter, run this script:
```bash
python lerobot/scripts/find_motors_bus_port.py
```
#### Example outputs of script

##### Mac:
Example output leader arm's port: `/dev/tty.usbmodem575E0031751`

```bash
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your MotorsBus and press Enter when done.

[...Disconnect leader arm and press Enter...]

The port of this MotorsBus is /dev/tty.usbmodem575E0031751
Reconnect the usb cable.
```

Example output follower arm port: `/dev/tty.usbmodem575E0032081`

```
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your MotorsBus and press Enter when done.

[...Disconnect follower arm and press Enter...]

The port of this MotorsBus is /dev/tty.usbmodem575E0032081
Reconnect the usb cable.
```

##### Linux:
On Linux, you might need to give access to the USB ports by running:
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

Example output leader arm port: `/dev/ttyACM0`

```bash
Finding all available ports for the MotorBus.
['/dev/ttyACM0', '/dev/ttyACM1']
Remove the usb cable from your MotorsBus and press Enter when done.

[...Disconnect leader arm and press Enter...]

The port of this MotorsBus is /dev/ttyACM0
Reconnect the usb cable.
```

Example output follower arm port: `/dev/ttyACM1`

```
Finding all available ports for the MotorBus.
['/dev/ttyACM0', '/dev/ttyACM1']
Remove the usb cable from your MotorsBus and press Enter when done.

[...Disconnect follower arm and press Enter...]

The port of this MotorsBus is /dev/ttyACM1
Reconnect the usb cable.
```

#### Update config file

Now that you have your ports, update the **port** default values of [`SO101RobotConfig`](https://github.com/huggingface/lerobot/blob/main/lerobot/common/robot_devices/robots/configs.py).
You will find a class called `so101` where you can update the `port` values with your actual motor ports:
```diff
@RobotConfig.register_subclass("so101")
@dataclass
class So101RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so101"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
-               port="/dev/tty.usbmodem58760431091",
+               port="{ADD YOUR LEADER PORT}",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
-                port="/dev/tty.usbmodem585A0076891",
+                port="{ADD YOUR FOLLOWER PORT}",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )
```

Here is a video of the process:

<video controls width="640" src="https://github.com/user-attachments/assets/fc45d756-31bb-4a61-b973-a87d633d08a7" type="video/mp4"></video>

### Set motor IDs

Now we need to set the motor ID for each motor. Plug your motor in only one of the two ports of the motor bus and run this script to set its ID to 1. Replace the text after --port to the corresponding control board port.
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58760432961 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

Then unplug your motor and plug the second motor and set its ID to 2.
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58760432961 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 2
```

Redo this process for all your motors until ID 6. Do the same for the 6 motors of the leader arm, but make sure to change the power supply if you use motors with different voltage.

Here is a video of the process:

<video controls width="640" src="https://github.com/user-attachments/assets/b31c115f-e706-4dcd-b7f1-4535da62416d" type="video/mp4"></video>

## Step-by-Step Assembly Instructions

The follower arm uses 6x STS3215 motors with 1/345 gearing. The leader however uses three differently geared motors to make sure it can both sustain its own weight and it can be moved without requiring much force. Which motor is needed for which joint is shown in table below.

| Leader-Arm Axis | Motor | Gear Ratio |
|-----------------|:-------:|:----------:|
| Base / Shoulder Yaw | 1 | 1 / 191 |
| Shoulder Pitch      | 2 | 1 / 345 |
| Elbow               | 3 | 1 / 191 |
| Wrist Roll          | 4 | 1 / 147 |
| Wrist Pitch         | 5 | 1 / 147 |
| Gripper             | 6 | 1 / 147 |


### Clean Parts
Remove all support material from the 3D-printed parts.

### Joint 1

- Place the first motor into the base.
- Fasten the motor with 4 M2x6mm screws (smallest screws). Two from the top and two from bottom.
- Slide over the first motor holder and fasten it using two M2x6mm screws (one on each side).
- Install both motor horns, securing the top horn with a M3x6mm screw.
- Attach the shoulder part.
- Tighten the shoulder part with 4 M3x6mm screws on top and 4 M3x6mm screws on the bottom
- Add the shoulder motor holder.

<video controls width="640" src="https://github.com/user-attachments/assets/b0ee9dee-a2d0-445b-8489-02ebecb3d639" type="video/mp4"></video>

### Joint 2

- Slide the second motor in from the top.
- Fasten the second motor with 4 M2x6mm screws.
- Attach both motor horns to motor 2, again use the M3x6mm horn screw.
- Attach the upper arm with 4 M3x6mm screws on each side.

<video controls width="640" src="https://github.com/user-attachments/assets/32453dc2-5006-4140-9f56-f0d78eae5155" type="video/mp4"></video>

### Joint 3

- Insert motor 3 and fasten using 4 M2x6mm screws
- Attach both motor horns to motor 3 and secure one again with a M3x6mm horn screw.
- Connect the forearm to motor 3 using 4 M3x6mm screws on each side.

<video controls width="640" src="https://github.com/user-attachments/assets/7384b9a7-a946-440c-b292-91391bcc4d6b" type="video/mp4"></video>

### Joint 4

- Slide over motor holder 4.
- Slide in motor 4.
- Fasten motor 4 with 4 M2x6mm screws and attach its motor horns, use a M3x6mm horn screw.

<video controls width="640" src="https://github.com/user-attachments/assets/dca78ad0-7c36-4bdf-8162-c9ac42a1506f" type="video/mp4"></video>

### Joint 5

- Insert motor 5 into the wrist holder and secure it with 2 M2x6mm front screws.
- Install only one motor horn on the wrist motor and secure it with a M3x6mm horn screw.
- Secure the wrist to motor 4 using 4 M3x6mm screws on both sides.

<video controls width="640" src="https://github.com/user-attachments/assets/55f5d245-976d-49ff-8b4a-59843c441b12" type="video/mp4"></video>

### Gripper / Handle

#### Follower:

- Attach the gripper to motor 5, attach it to the motor horn on the wrist using 4 M3x6mm screws.
- Insert the gripper motor and secure it with 2 M2x6mm screws on each side.
- Attach the motor horns and again use a M3x6mm horn screw.
- Install the gripper claw and secure it with 4 M3x6mm screws on both sides.

<video controls width="640" src="https://github.com/user-attachments/assets/6f766aa9-cfae-4388-89e7-0247f198c086" type="video/mp4"></video>

#### Leader:

- Mount the leader holder onto the wrist and secure it with 4 M3x6mm screws.
- Attach the handle to motor 5 using 1 M2x6mm screw.
- Insert the gripper motor, secure it with 2 M2x6mm screws on each side, attach a motor horn using a M3x6mm horn screw.
- Attach the follower trigger with 4 M3x6mm screws.

<video controls width="640" src="https://github.com/user-attachments/assets/1308c93d-2ef1-4560-8e93-a3812568a202" type="video/mp4"></video>

##### Wiring

- Attach the motor controller on the back.
- Then insert all wires, use the wire guides everywhere to make sure the wires don't unplug themselves and stay in place.

<video controls width="640" src="https://github.com/user-attachments/assets/4c2cacfd-9276-4ee4-8bf2-ba2492667b78" type="video/mp4"></video>

## Calibrate

Next, you'll need to calibrate your SO-101 robot to ensure that the leader and follower arms have the same position values when they are in the same physical position.
The calibration process is very important because it allows a neural network trained on one SO-101 robot to work on another.

#### Manual calibration of follower arm

You will need to move the follower arm to these positions sequentially, note that the rotated position is on the right side of the robot and you have to open the gripper fully.

| 1. Middle position | 2. Zero position                                                                                                                                       | 3. Rotated position                                                                                                                                             | 4. Rest position                                                                                                                                       |
| ------------ |------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="../media/so101/follower_middle.webp?raw=true" alt="SO-101 leader arm middle position" title="SO-101 leader arm middle position" style="width:100%;"> | <img src="../media/so101/follower_zero.webp?raw=true" alt="SO-101 leader arm zero position" title="SO-101 leader arm zero position" style="width:100%;"> | <img src="../media/so101/follower_rotated.webp?raw=true" alt="SO-101 leader arm rotated position" title="SO-101 leader arm rotated position" style="width:100%;"> | <img src="../media/so101/follower_rest.webp?raw=true" alt="SO-101 leader arm rest position" title="SO-101 leader arm rest position" style="width:100%;"> |

Make sure both arms are connected and run this script to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

#### Manual calibration of leader arm
You will also need to move the leader arm to these positions sequentially:

| 1. Middle position | 2. Zero position                                                                                                                                       | 3. Rotated position                                                                                                                                             | 4. Rest position                                                                                                                                       |
| ------------ |------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="../media/so101/leader_middle.webp?raw=true" alt="SO-101 leader arm middle position" title="SO-101 leader arm middle position" style="width:100%;"> | <img src="../media/so101/leader_zero.webp?raw=true" alt="SO-101 leader arm zero position" title="SO-101 leader arm zero position" style="width:100%;"> | <img src="../media/so101/leader_rotated.webp?raw=true" alt="SO-101 leader arm rotated position" title="SO-101 leader arm rotated position" style="width:100%;"> | <img src="../media/so101/leader_rest.webp?raw=true" alt="SO-101 leader arm rest position" title="SO-101 leader arm rest position" style="width:100%;"> |

Run this script to launch manual calibration:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```
## Control your robot

Congrats 🎉, your robot is all set to learn a task on its own. Next we will explain to you how to train a neural network to autonomously control a real robot.

**You'll learn to:**
1. How to record and visualize your dataset.
2. How to train a policy using your data and prepare it for evaluation.
3. How to evaluate your policy and visualize the results.

By following these steps, you'll be able to replicate tasks like picking up a Lego block and placing it in a bin with a high success rate, as demonstrated in [this video](https://x.com/RemiCadene/status/1814680760592572934).

This tutorial is specifically made for the affordable [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot, but it contains additional information to be easily adapted to various types of robots like [Aloha bimanual robot](https://aloha-2.github.io) by changing some configurations. The SO-101 consists of a leader arm and a follower arm, each with 6 motors. It can work with one or several cameras to record the scene, which serve as visual sensors for the robot.

During the data collection phase, you will control the follower arm by moving the leader arm. This process is known as "teleoperation." This technique is used to collect robot trajectories. Afterward, you'll train a neural network to imitate these trajectories and deploy the network to enable your robot to operate autonomously.

If you encounter any issues at any step of the tutorial, feel free to seek help on [Discord](https://discord.com/invite/s3KuuzsPFb) or don't hesitate to iterate with us on the tutorial by creating issues or pull requests.

## Teleoperate

Run this simple script to teleoperate your robot (it won't connect and display the cameras):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

The teleoperate command will automatically:
1. Identify any missing calibrations and initiate the calibration procedure.
2. Connect the robot and start teleoperation.

## Setup Cameras

To connect a camera you have three options:
1. OpenCVCamera which allows us to use any camera: usb, realsense, laptop webcam
2. iPhone camera with MacOS
3. Phone camera on Linux

### Use OpenCVCamera

The [`OpenCVCamera`](../lerobot/common/robot_devices/cameras/opencv.py) class allows you to efficiently record frames from most cameras using the [`opencv2`](https://docs.opencv.org) library.  For more details on compatibility, see [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

To instantiate an [`OpenCVCamera`](../lerobot/common/robot_devices/cameras/opencv.py), you need a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera like a webcam of a laptop, the camera index is usually `0` but it might differ, and the camera index might change if you reboot your computer or re-plug your camera. This behavior depends on your operating system.

To find the camera indices, run the following utility script, which will save a few frames from each detected camera:
```bash
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```

The output will look something like this if you have two cameras connected:
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
Frame: 0000 Latency (ms): 39.52
[...]
Frame: 0046 Latency (ms): 40.07
Images have been saved to outputs/images_from_opencv_cameras
```

Check the saved images in `outputs/images_from_opencv_cameras` to identify which camera index corresponds to which physical camera (e.g. `0` for `camera_00` or `1` for `camera_01`):
```
camera_00_frame_000000.png
[...]
camera_00_frame_000047.png
camera_01_frame_000000.png
[...]
camera_01_frame_000047.png
```

Note: Some cameras may take a few seconds to warm up, and the first frame might be black or green.

Now that you have the camera indexes, you should change them in the config. You can also change the fps, width or height of the camera.

The camera config is defined per robot, can be found here [`RobotConfig`](https://github.com/huggingface/lerobot/blob/main/lerobot/common/robot_devices/robots/configs.py) and looks like this:
```python
cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": OpenCVCameraConfig(
                camera_index=0,  <-- UPDATE HERE
                fps=30,
                width=640,
                height=480,
            ),
            "base": OpenCVCameraConfig(
                camera_index=1,   <-- UPDATE HERE
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
```

### Use your phone
#### Mac:

To use your iPhone as a camera on macOS, enable the Continuity Camera feature:
- Ensure your Mac is running macOS 13 or later, and your iPhone is on iOS 16 or later.
- Sign in both devices with the same Apple ID.
- Connect your devices with a USB cable or turn on Wi-Fi and Bluetooth for a wireless connection.

For more details, visit [Apple support](https://support.apple.com/en-gb/guide/mac-help/mchl77879b8a/mac).

Your iPhone should be detected automatically when running the camera setup script in the next section.

#### Linux:

If you want to use your phone as a camera on Linux, follow these steps to set up a virtual camera

1. *Install `v4l2loopback-dkms` and `v4l-utils`*. Those packages are required to create virtual camera devices (`v4l2loopback`) and verify their settings with the `v4l2-ctl` utility from `v4l-utils`. Install them using:
```python
sudo apt install v4l2loopback-dkms v4l-utils
```
2. *Install [DroidCam](https://droidcam.app) on your phone*. This app is available for both iOS and Android.
3. *Install [OBS Studio](https://obsproject.com)*. This software will help you manage the camera feed. Install it using [Flatpak](https://flatpak.org):
```python
flatpak install flathub com.obsproject.Studio
```
4. *Install the DroidCam OBS plugin*. This plugin integrates DroidCam with OBS Studio. Install it with:
```python
flatpak install flathub com.obsproject.Studio.Plugin.DroidCam
```
5. *Start OBS Studio*. Launch with:
```python
flatpak run com.obsproject.Studio
```
6. *Add your phone as a source*. Follow the instructions [here](https://droidcam.app/obs/usage). Be sure to set the resolution to `640x480`.
7. *Adjust resolution settings*. In OBS Studio, go to `File > Settings > Video`. Change the `Base(Canvas) Resolution` and the `Output(Scaled) Resolution` to `640x480` by manually typing it in.
8. *Start virtual camera*. In OBS Studio, follow the instructions [here](https://obsproject.com/kb/virtual-camera-guide).
9. *Verify the virtual camera setup*. Use `v4l2-ctl` to list the devices:
```python
v4l2-ctl --list-devices
```
You should see an entry like:
```
VirtualCam (platform:v4l2loopback-000):
/dev/video1
```
10. *Check the camera resolution*. Use `v4l2-ctl` to ensure that the virtual camera output resolution is `640x480`. Change `/dev/video1` to the port of your virtual camera from the output of `v4l2-ctl --list-devices`.
```python
v4l2-ctl -d /dev/video1 --get-fmt-video
```
You should see an entry like:
```
>>> Format Video Capture:
>>> Width/Height      : 640/480
>>> Pixel Format      : 'YUYV' (YUYV 4:2:2)
```

Troubleshooting: If the resolution is not correct you will have to delete the Virtual Camera port and try again as it cannot be changed.

If everything is set up correctly, you can proceed with the rest of the tutorial.

### Add wrist camera
If you have an additional camera you can add a wrist camera to the SO101. There are already many premade wrist camera holders that you can find in the SO101 repo: [Wrist camera's](https://github.com/TheRobotStudio/SO-ARM100#wrist-cameras)

## Teleoperate with cameras

We can now teleoperate again while at the same time visualizing the cameras and joint positions with `rerun`.

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true
```

## Record a dataset

Once you're familiar with teleoperation, you can record your first dataset with SO-101.

We use the Hugging Face hub features for uploading your dataset. If you haven't previously used the Hub, make sure you can login via the cli using a write-access token, this token can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens).

Add your token to the cli by running this command:
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Then store your Hugging Face repository name in a variable:
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

Now you can record a dataset, to record 2 episodes and upload your dataset to the hub execute this command:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/so101_test \
  --control.tags='["so101","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.display_data=true \
  --control.push_to_hub=true
```

You will see a lot of lines appearing like this one:
```
INFO 2024-08-10 15:02:58 ol_robot.py:219 dt:33.34 (30.0hz) dtRlead: 5.06 (197.5hz) dtWfoll: 0.25 (3963.7hz) dtRfoll: 6.22 (160.7hz) dtRlaptop: 32.57 (30.7hz) dtRphone: 33.84 (29.5hz)
```
It contains:
- `2024-08-10 15:02:58` which is the date and time of the call to the print function,
- `ol_robot.py:219` which is the end of the file name and the line number where the print function is called  (`lerobot/scripts/control_robot.py` line `219`).
- `dt:33.34 (30.0hz)` which is the "delta time" or the number of milliseconds spent between the previous call to `robot.teleop_step(record_data=True)` and the current one, associated with the frequency (33.34 ms equals 30.0 Hz) ; note that we use `--fps 30` so we expect 30.0 Hz ; when a step takes more time, the line appears in yellow.
- `dtRlead: 5.06 (197.5hz)` which is the delta time of reading the present position of the leader arm.
- `dtWfoll: 0.25 (3963.7hz)` which is the delta time of writing the goal position on the follower arm ; writing is asynchronous so it takes less time than reading.
- `dtRfoll: 6.22 (160.7hz)` which is the delta time of reading the present position on the follower arm.
- `dtRlaptop:32.57 (30.7hz) ` which is the delta time of capturing an image from the laptop camera in the thread running asynchronously.
- `dtRphone:33.84 (29.5hz)` which is the delta time of capturing an image from the phone camera in the thread running asynchronously.

#### Dataset upload
Locally your dataset is stored in this folder: `~/.cache/huggingface/lerobot/{repo-id}` (e.g. `data/cadene/so101_test`). At the end of data recording, your dataset will be uploaded on your Hugging Face page (e.g. https://huggingface.co/datasets/cadene/so101_test) that you can obtain by running:
```bash
echo https://huggingface.co/datasets/${HF_USER}/so101_test
```
Your dataset will be automatically tagged with `LeRobot` for the community to find it easily, and you can also add custom tags (in this case `tutorial` for example).

You can look for other LeRobot datasets on the hub by searching for `LeRobot` [tags](https://huggingface.co/datasets?other=LeRobot).

#### Record function

The `record` function provides a suite of tools for capturing and managing data during robot operation:
1. Set the flow of data recording using command line arguments:
   - `--control.warmup_time_s=10` defines the number of seconds before starting data collection. It allows the robot devices to warmup and synchronize (10 seconds by default).
   - `--control.episode_time_s=60` defines the number of seconds for data recording for each episode (60 seconds by default).
   - `--control.reset_time_s=60` defines the number of seconds for resetting the environment after each episode (60 seconds by default).
   - `--control.num_episodes=50` defines the number of episodes to record (50 by default).
2. Control the flow during data recording using keyboard keys:
   - Press right arrow `->` at any time during episode recording to early stop and go to resetting. Same during resetting, to early stop and to go to the next episode recording.
   - Press left arrow `<-` at any time during episode recording or resetting to early stop, cancel the current episode, and re-record it.
   - Press escape `ESC` at any time during episode recording to end the session early and go straight to video encoding and dataset uploading.
3. Checkpoints are done set during recording, so if any issue occurs, you can resume recording by re-running the same command again with `--control.resume=true`. You will need to manually delete the dataset directory if you want to start recording from scratch.

#### Tips for gathering data

Once you're comfortable with data recording, you can create a larger dataset for training. A good starting task is grasping an object at different locations and placing it in a bin. We suggest recording at least 50 episodes, with 10 episodes per location. Keep the cameras fixed and maintain consistent grasping behavior throughout the recordings. Also make sure the object you are manipulating is visible on the camera's. A good rule of thumb is you should be able to do the task yourself by only looking at the camera images.

In the following sections, you’ll train your neural network. After achieving reliable grasping performance, you can start introducing more variations during data collection, such as additional grasp locations, different grasping techniques, and altering camera positions.

Avoid adding too much variation too quickly, as it may hinder your results.

#### Troubleshooting:
- On Linux, if the left and right arrow keys and escape key don't have any effect during data recording, make sure you've set the `$DISPLAY` environment variable. See [pynput limitations](https://pynput.readthedocs.io/en/latest/limitations.html#linux).

## Visualize a dataset

If you uploaded your dataset to the hub with `--control.push_to_hub=true`, you can [visualize your dataset online](https://huggingface.co/spaces/lerobot/visualize_dataset) by copy pasting your repo id given by:
```bash
echo ${HF_USER}/so101_test
```

If you didn't upload with `--control.push_to_hub=false`, you can visualize it locally with (via a window in the browser `http://127.0.0.1:9090` with the visualization tool):
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/so101_test \
  --local-files-only 1
```

This will launch a local web server that looks like this:

<div style="text-align:center;">
  <img src="../media/tutorial/visualize_dataset_html.webp?raw=true" alt="Koch v1.1 leader and follower arms" title="Koch v1.1 leader and follower arms" width="100%"></img>
</div>

## Replay an episode

A useful feature is the `replay` function, which allows to replay on your robot any episode that you've recorded or episodes from any dataset out there. This function helps you test the repeatability of your robot's actions and assess transferability across robots of the same model.

You can replay the first episode on your robot with:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/so101_test \
  --control.episode=0
```

Your robot should replicate movements similar to those you recorded. For example, check out [this video](https://x.com/RemiCadene/status/1793654950905680090) where we use `replay` on a Aloha robot from [Trossen Robotics](https://www.trossenrobotics.com).

## Train a policy

To train a policy to control your robot, use the [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) script. A few arguments are required. Here is an example command:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true
```

Let's explain the command:
1. We provided the dataset as argument with `--dataset.repo_id=${HF_USER}/so101_test`.
2. We provided the policy with `policy.type=act`. This loads configurations from [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py). Importantly, this policy will automatically adapt to the number of motor states, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.

Training should take several hours. You will find checkpoints in `outputs/train/act_so101_test/checkpoints`.

To resume training from a checkpoint, below is an example command to resume from `last` checkpoint of the `act_so101_test` policy:
```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

#### Upload policy checkpoints

Once training is done, upload the latest checkpoint with:
```bash
huggingface-cli upload ${HF_USER}/act_so101_test \
  outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

You can also upload intermediate checkpoints with:
```bash
CKPT=010000
huggingface-cli upload ${HF_USER}/act_so101_test${CKPT} \
  outputs/train/act_so101_test/checkpoints/${CKPT}/pretrained_model
```

## Evaluate your policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_so101_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with  (e.g. `outputs/train/eval_act_so101_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/act_so101_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_act_so101_test`).
