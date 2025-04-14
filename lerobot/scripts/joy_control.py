from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.feetech import TorqueMode 

import time

import evdev
from evdev import InputDevice, categorize, ecodes

import argparse
import logging
logger = logging.getLogger("stadia_control")


follower_config = FeetechMotorsBusConfig(
    port="/dev/ttyUSB0",
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"),
        "elbow_flex": (3, "sts3215"),
        "wrist_flex": (4, "sts3215"),
        "wrist_roll": (5, "sts3215"),
        "gripper": (6, "sts3215"),
    },
)

follower_arm = FeetechMotorsBus(follower_config)

def joy_loop(device):
    logger.info("[***] Connecting to motors... (port: " + str(follower_config.port) + ")")
    follower_arm.connect()
    logger.debug("[***] Connected to motors")

    follower_pos = follower_arm.read("Present_Position")
    logger.debug("[***] Present position: " + str(follower_pos))

    home_pos = follower_pos.copy()
    logger.debug("[***] Home position: " + str(home_pos))

    follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)
    logger.debug("[***] Torque enabled")
    
    left_BUMP = False
    right_BUMP = False
    left_TR = False
    right_TR = False

    recordings_state = False
    # cretae file to save the recordings, name of file replay_<timestamp>.txt
    session_record = open("replay_" + str(int(time.time())) + ".txt", "w")
    logger.info("[***] Recordings file: [" + session_record.name + "] created")

    update_flag = False
    movement_delta = 10
    stick_stresshold = 30
    stick_center = 128
    replay_speed = 0.03
    try:
 
        for event in device.read_loop():
            #print(event)
            if event.type == ecodes.EV_KEY:
                #print(categorize(event))
                keyevent = categorize(event)
                if keyevent.event.code == ecodes.BTN_TR:
                    if keyevent.event.value == 1:
                        right_BUMP = True
                    elif keyevent.event.value == 0:
                        right_BUMP = False
                elif keyevent.event.code == ecodes.BTN_TL:
                    if keyevent.event.value == 1:
                        left_BUMP = True
                    elif keyevent.event.value == 0:
                        left_BUMP = False
                elif keyevent.event.code == ecodes.BTN_TRIGGER_HAPPY4:
                    if keyevent.event.value == 1:
                        left_TR = True
                    elif keyevent.event.value == 0:
                        left_TR = False
                elif keyevent.event.code == ecodes.BTN_TRIGGER_HAPPY3:
                    if keyevent.event.value == 1:
                        right_TR = True
                    elif keyevent.event.value == 0:
                        right_TR = False
                elif keyevent.event.code == ecodes.BTN_START:
                    if keyevent.event.value == 1:
                        logger.info("[***] Disconnecting from motors...")
                        break
                elif keyevent.event.code == ecodes.BTN_Y:
                    if keyevent.event.value == 1:
                        logger.info("[***] Go to home position...")
                        follower_pos = home_pos.copy()
                elif keyevent.event.code == ecodes.BTN_B:
                    if keyevent.event.value == 1:
                        logger.info("[***] Check path to home position...")
                        follower_arm.write("Goal_Position", home_pos)
                        time.sleep(1)
                elif keyevent.event.code == ecodes.BTN_X:
                    if keyevent.event.value == 1:
                        logger.info("[***] Save current position...")
                        home_pos = follower_pos.copy()
                elif keyevent.event.code == ecodes.BTN_A:
                    if keyevent.event.value == 1:
                        logger.info("[***] Current position: " + str(follower_pos))
                        session_record.write(str(follower_pos) + "\n")
                        session_record.flush()
 
                elif keyevent.event.code == ecodes.BTN_TRIGGER_HAPPY2:
                    if keyevent.event.value == 1:
                        if recordings_state:
                            logger.info("[***] Pause recording")
                            recordings_state = False
                        else:
                            logger.info("[***] Start recording")
                            recordings_state = True
                elif keyevent.event.code == ecodes.BTN_TRIGGER_HAPPY1:
                    if keyevent.event.value == 1:
                        logger.info("[***] Replay current session")
                        with open(session_record.name, "r") as f:
                            for line in f:
                                # repmove [ and ] from line
                                line = line.strip().replace("[", "").replace("]", "")
                                pos = line.strip().split(" ")
                                logger.debug(str(pos))
                                pos = [int(x) for x in pos]
                                follower_arm.write("Goal_Position", pos)
                                time.sleep(replay_speed)

            elif event.type == ecodes.EV_ABS:
                #print(categorize(event))
                absevent = categorize(event)
                if absevent.event.code == ecodes.ABS_X:
                    update_flag = True
                    joint = 0
                    if absevent.event.value > stick_center + stick_stresshold:
                        follower_pos[joint] += movement_delta
                    elif absevent.event.value < stick_center - stick_stresshold:
                        follower_pos[joint] -= movement_delta

                elif absevent.event.code == ecodes.ABS_Y:
                    update_flag = True
                    joint_1 = 1
                    joint_2 = 2
                    if absevent.event.value > stick_center + stick_stresshold:
                        if not left_TR:
                            follower_pos[joint_1] -= movement_delta
                        if not left_BUMP:
                            follower_pos[joint_2] += movement_delta
                    elif absevent.event.value < stick_center - stick_stresshold:
                        if not left_TR:
                            follower_pos[joint_1] += movement_delta
                        if not left_BUMP:
                            follower_pos[joint_2] -= movement_delta
                elif absevent.event.code == ecodes.ABS_Z:
                    update_flag = True
                    joint = 4
                    if absevent.event.value > stick_center + stick_stresshold:
                        follower_pos[joint] -= movement_delta
                    elif absevent.event.value < stick_center - stick_stresshold:
                        follower_pos[joint] += movement_delta
                elif absevent.event.code == ecodes.ABS_RZ:
                    update_flag = True
                    joint = 3
                    if absevent.event.value > stick_center + stick_stresshold:
                        follower_pos[joint] += movement_delta
                    elif absevent.event.value < stick_center - stick_stresshold:
                        follower_pos[joint] -= movement_delta
                elif absevent.event.code == ecodes.ABS_RX:
                    pass
                elif absevent.event.code == ecodes.ABS_RY:
                    pass
                elif absevent.event.code == ecodes.ABS_HAT0X:
                    if absevent.event.value == 1:
                        movement_delta += 1
                    elif absevent.event.value == -1:
                        movement_delta -= 1
                    logger.info("[***] Movement delta: " + str(movement_delta))
                elif absevent.event.code == ecodes.ABS_HAT0Y:
                    if absevent.event.value == 1:
                        replay_speed += 0.01
                    elif absevent.event.value == -1:
                        replay_speed -= 0.01
                    logger.info("[***] Replay speed: " + str(replay_speed))
            
            if right_TR:
                update_flag = True
                follower_pos[5] = home_pos[5] + 1150
            else:
                update_flag = True
                follower_pos[5] = home_pos[5]

            if update_flag:
                update_flag = False
                if recordings_state:
                    # write the current position to the file
                    session_record.write(str(follower_pos) + "\n")
                    session_record.flush()
                    follower_last_pos = follower_pos.copy()
                    logger.debug("[***] Recording position: " + str(follower_pos))

                follower_arm.write("Goal_Position", follower_pos)

    except KeyboardInterrupt:
        pass

    logger.info("[***] Go to home position...")
    follower_arm.write("Goal_Position", home_pos)
    time.sleep(1)
    follower_arm.disconnect()
    logger.info("[***] Disconnected from motors")

    logger.info("[***] Closing session record file {" + session_record.name + "}")
    session_record.close()

    return None

def find_stadia_device(count=20):
    for i in range(0, count):
        try:
            device = evdev.InputDevice(f'/dev/input/event{i}')
            logger.debug(str(device.name))
            if str(device.name).startswith("Stadia"):
                logger.info(f"Found and use device: event{i}")
                return device
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug(f"Error accessing /dev/input/event{i}")

    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Stadia joystick control script")
    parser.add_argument('-c', '--check', type=int, help='Count for check evdev devices')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-U', '--uart', type=str, help='Set UART port for robot servos')
    parser.add_argument('-S', '--stadia', type=str, help='Set Stadia device')
    parser.add_argument('-r', '--replay', type=str, help='Just replay the session file (dont use joystick)')
    parser.add_argument('-s', '--speed', type=float, help='Set replay speed (default: 1.00)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.uart:
        follower_config.port = args.uart
        logger.info(f"Using UART port: {args.uart}")
    else:
        logger.info(f"Using default UART port: {follower_config.port}")

    if args.replay:
        logger.info(f"Replay session file: {args.replay}")
        logger.info(f"Replay speed: {args.speed}")

        logger.info("[***] Connecting to motors... (port: " + str(follower_config.port) + ")")
        follower_arm.connect()
        logger.debug("[***] Connected to motors")

        follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)
        logger.debug("[***] Torque enabled")
 
        try:
            while True:
                with open(args.replay, "r") as f:
                    for line in f:
                        # repmove [ and ] from line
                        line = line.strip().replace("[", "").replace("]", "")
                        pos = line.strip().split(" ")
                        logger.debug(str(pos))
                        pos = [int(x) for x in pos]
                        follower_arm.write("Goal_Position", pos)
                        time.sleep(args.speed if args.speed else 1.00)
                logger.info("Press Ctrl+C to exit replay")
                logger.info("Press any key to play again")
                # wait for any input
                input()
        except KeyboardInterrupt:
            pass
    
        follower_arm.disconnect()
        exit(0)

    device = None
    if args.stadia:
        try:
            device = evdev.InputDevice(args.stadia)
            logger.info(f"Using stadia device: {args.stadia}")
        except FileNotFoundError:
            logger.error(f"Stadia device not found: {args.stadia}")
            exit(1)
        except OSError:
            logger.error(f"Error accessing stadia device: {args.stadia}")
            exit(1)
    else:
        logger.info("Searching for stadia device...")
        count = 20
        if args.check:
            count = args.check
        device = find_stadia_device(count)
        if device is None:
            logger.error("No stadia device found")
            exit(1)

    description = """
    
    Button Layout (Stadia Controller):
    
        -- Movements --
    [LEFT_STICK] - Control joints 0 1 and 2
    [L2] - Block joint 2
    [L1] - Block joint 1
    [Right_STICK] - Control joints 3 and 4
    [R2] - Open/Close gripper
    [DPAD left/right] - Adjust movement delta
    
        -- Session record -- 
    [CAPTURE] - Start/Pause recording session file
    [ASSIST] - Replay current session
    [DPAD up/down] - Adjust replay speed
    [A] - Print current position and save it to session file
    
        -- Actions --
    [X] - Save current position as home position
    [Y] - Move to home position
    [B] - Move to home position and return to current position
    [A] - Print current position and save it to session file
    
        -- Status --
    [MENU] - Exit

    """
    logger.info(description)
    layout = """
          O  <-- End Effector (Gripper)
          |
       [Joint 5] <-- R2 (Gripper open/close)
          |
       [Joint 4] <-- Right Stick horizontal movement
          |
       [Joint 3] <-- Right Stick vertical movement 
          |
       [Joint 2] <-- Left Stick vertical movement ( press and hold L1 to block position )
          |
       [Joint 1] <-- Left Stick vertical movement ( press and hold L2 to block position )
          |
       [Joint 0] <-- Left Stick horizontal movement
          |
         [Base]
    """
    logger.info(layout)

    joy_loop(device)

