This tutorial explains how to use [Moss v1](https://github.com/jess-moss/moss-robot-arms) with LeRobot.

## Sourcing a MyArm

A MyArm can be purchased at the [Elephant Robotics M&C Store Section](https://shop.elephantrobotics.com/collections/myarm-mc)

**Important** Make sure your MyArmM and MyArmC firmware is up to date. The
firmware can be updated using the MyStudio app.

## Install LeRobot

This tutorial was tested on a Linux machine.

In the LeRobot repository, install the dependencies including the `myarm` extra:
```bash
poetry install --extras my_arm
```

## Verify Teleop Without LeRobot First

1. Plug in both the MyArm M and MyArm C via USB to your computer.
Ensure the following:
- both power supplies are plugged in
- the e-stop is disabled
- the MyArmM power button is pressed

**IMPORTANT:**
If you try to connect and it fails, then you find out it was an e-stop or other power issue, you will NEED to unplug the USB, restart the robot, fix the power issue, and then plug the USB back in.

2. Then, in both `MyArmM` and `MyArmC` interfaces select "Transponder -> USB UART".

**Find USB ports associated to your arms**

The `acton_ai` library will automatically scan for likely arms and connect
to them. It will automatically identify which is the Mover and which is the Controller.

You can also hardcode the ports in the `lerobot/configs/robot/myarm.yaml` file.

If you see exceptions relating to finding or connecting to arms, **read them**.

The acton_ai library will provide descriptive error messages if it cannot find or connect to the arms.

Troubleshooting: On Linux, you might need to give access to the USB ports by running:
```bash
sudo chmod a+rw /dev/ttyACM*
```

3. Finally, run the teleop script to verify that you can control the arms:
```bash
acton_teleop
```

Move the leader around and verify that the follower arm follows, and is accurate.

If anything seems amis, follow the calibrate section below. This script will
log debug issues relating to joints out of bounds, and other problems that might occur.
Pay attention to the logs and fix any issues before proceeding.

## Calibrate

If anything seems amiss, run
```bash
acton_calibrate
```

and follow directions in the console. This script is always being maintained, please add
issues on the `acton_ai` github so we can improve it.

## Teleoperate

**Simple teleop**
Then you are ready to teleoperate your robot via lerobot! Run this simple script (it won't connect and display the cameras):
```bash
python3 lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/myarm.yaml \
    --robot-overrides '~cameras' \
    --display-cameras 0
```


**Teleop with displaying cameras**
Follow [this guide to setup your cameras](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#c-add-your-cameras-with-opencvcamera). Then you will be able to display the cameras on your computer while you are teleoperating by running the following code. This is useful to prepare your setup before recording your first dataset.
```bash
python3 lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/myarm.yaml
```

# What Next?

Well, now your robot is compatible with lerobot! Check out other tutorials for how to train a policy, visualize a dataset, train a model, and execute a model policy using a robot!

You can start out by checking out the other commands `control_robot` supports using the `--help` menu, such as `record`!
