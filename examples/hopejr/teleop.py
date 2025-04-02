from follower import HopeJuniorRobot
from leader import (
    HomonculusArm,
    HomonculusGlove,
    EncoderReader
)
from visualizer import value_to_color

import time
import numpy as np

import pickle
import pygame
import typer

def main(
    calibrate_glove: bool = typer.Option(False, "--calibrate-glove", help="Calibrate the glove"),
    calibrate_exoskeleton: bool = typer.Option(False, "--calibrate-exoskeleton", help="Calibrate the exoskeleton"),
    freeze_fingers: bool = typer.Option(False, "--freeze-fingers", help="Freeze the fingers"),
    freeze_arm: bool = typer.Option(False, "--freeze-arm", help="Freeze the arm")):
    show_loads: bool = typer.Option(False, "--show-loads", help="Show the loads in a GUI")
    robot = HopeJuniorRobot()
    

    robot.connect_hand()
    #robot.connect_arm() 
    #read pos
    print(robot.hand_bus.read("Present_Position"))
    #print(robot.arm_bus.read("Present_Position", "shoulder_pitch"))
    #print(robot.arm_bus.read("Present_Position",["shoulder_yaw","shoulder_roll","elbow_flex","wrist_roll","wrist_yaw","wrist_pitch"]))

    #for i in range(10):
    #    time.sleep(0.1)
    #    robot.apply_arm_config('examples/hopejr/settings/config.yaml')

    # #calibrate arm
    #arm_calibration = robot.get_arm_calibration()
    #exoskeleton = HomonculusArm(serial_port="/dev/ttyACM2")
    #robot.arm_bus.write("Goal_Position", robot.arm_calib_dict["start_pos"][0]*0.7 + robot.arm_calib_dict["end_pos"][0]*0.3, ["shoulder_pitch"])
    
    #if calibrate_exoskeleton:   
    #    exoskeleton.run_calibration(robot)

    #file_path = "examples/hopejr/settings/arm_calib.pkl"
    #with open(file_path, "rb") as f:
    #   calib_dict = pickle.load(f)
    #print("Loaded dictionary:", calib_dict)
    #exoskeleton.set_calibration(calib_dict)

    #calibrate hand
    hand_calibration = robot.get_hand_calibration()
    glove = HomonculusGlove(serial_port = "/dev/ttyACM1")

    if calibrate_glove:
        glove.run_calibration()
        
    file_path = "examples/hopejr/settings/hand_calib.pkl"
    with open(file_path, "rb") as f:
         calib_dict = pickle.load(f)
    print("Loaded dictionary:", calib_dict)
    glove.set_calibration(calib_dict)

    robot.hand_bus.set_calibration(hand_calibration)
    #robot.arm_bus.set_calibration(arm_calibration)

    # Initialize Pygame
    # pygame.init()

    # # Set up the display
    # screen = pygame.display.set_mode((800, 600))

    # pygame.display.set_caption("Robot Hand Visualization")


    # # Create hand structure with 16 squares and initial values
    # hand_components = []

    # # Add thumb (4 squares in diamond shape)
    # thumb_positions = [
    #     (150, 300), (125, 350),
    #     (175, 350), (150, 400)
    # ]
    # for pos in thumb_positions:
    #     hand_components.append({"pos": pos, "value": 0})

    # # Add fingers (4 fingers with 3 squares each in vertical lines)
    # finger_positions = [
    #     (200, 100),  # Index
    #     (250, 100),  # Middle
    #     (300, 100),  # Ring
    #     (350, 100)   # Pinky
    # ]

    # for x, y in finger_positions:
    #     for i in range(3):
    #         hand_components.append({"pos": (x, y + i * 50), "value": 0})

    for i in range(1000000000000000):
            # robot.apply_arm_config('examples/hopejr/settings//config.yaml')
            # robot.arm_bus.write("Acceleration", 50, "shoulder_yaw")
            # joint_names = ["shoulder_pitch", "shoulder_yaw", "shoulder_roll", "elbow_flex", "wrist_roll", "wrist_yaw", "wrist_pitch"]
            # joint_values = exoskeleton.read_running_average(motor_names=joint_names, linearize=True)

            # joint_values = joint_values.round().astype(int)
            # joint_dict = {k: v for k, v in zip(joint_names, joint_values, strict=False)}

            # motor_values = []
            # motor_names = []
            # motor_names += ["shoulder_pitch", "shoulder_yaw", "shoulder_roll", "elbow_flex", "wrist_roll", "wrist_yaw", "wrist_pitch"]
            # motor_values += [joint_dict[name]-30 for name in motor_names]

            # motor_values = np.array(motor_values)
            # motor_values = np.clip(motor_values, 0, 100)
            # if not freeze_arm:
            #     robot.arm_bus.write("Goal_Position", motor_values, motor_names)

            if not freeze_fingers:#include hand
                hand_joint_names = []
                hand_joint_names += ["thumb_3", "thumb_2", "thumb_1", "thumb_0"]#, "thumb_3"]
                hand_joint_names += ["index_0", "index_1", "index_2"]
                hand_joint_names += ["middle_0", "middle_1", "middle_2"]
                hand_joint_names += ["ring_0", "ring_1", "ring_2"]
                hand_joint_names += ["pinky_0", "pinky_1", "pinky_2"]
                hand_joint_values = glove.read(hand_joint_names)
                hand_joint_values = hand_joint_values.round( ).astype(int)
                hand_joint_dict = {k: v for k, v in zip(hand_joint_names, hand_joint_values, strict=False)}

                hand_motor_values = []
                hand_motor_names = []

                # Thumb
                hand_motor_names += ["thumb_basel_rotation", "thumb_mcp", "thumb_pip", "thumb_dip"]#, "thumb_MCP"]
                hand_motor_values += [
                    hand_joint_dict["thumb_3"],
                    hand_joint_dict["thumb_2"],
                    hand_joint_dict["thumb_1"],
                    hand_joint_dict["thumb_0"]
                ]

                # # Index finger
                index_splay = 0.1
                hand_motor_names += ["index_flexor", "index_pinky_side", "index_thumb_side"]
                hand_motor_values += [
                    hand_joint_dict["index_2"],
                    (100 - hand_joint_dict["index_0"]) * index_splay + hand_joint_dict["index_1"] * (1 - index_splay),
                    hand_joint_dict["index_0"] * index_splay + hand_joint_dict["index_1"] * (1 - index_splay),
                ]

                # Middle finger
                middle_splay = 0.1
                hand_motor_names += ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
                hand_motor_values += [
                    hand_joint_dict["middle_2"],
                    hand_joint_dict["middle_0"] * middle_splay + hand_joint_dict["middle_1"] * (1 - middle_splay),
                    (100 - hand_joint_dict["middle_0"]) * middle_splay + hand_joint_dict["middle_1"] * (1 - middle_splay),
                ]

                # # Ring finger
                ring_splay = 0.1
                hand_motor_names += ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
                hand_motor_values += [
                    hand_joint_dict["ring_2"],
                    (100 - hand_joint_dict["ring_0"]) * ring_splay + hand_joint_dict["ring_1"] * (1 - ring_splay),
                    hand_joint_dict["ring_0"] * ring_splay + hand_joint_dict["ring_1"] * (1 - ring_splay),
                ]

                # # Pinky finger
                pinky_splay = -.1
                hand_motor_names += ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]
                hand_motor_values += [
                    hand_joint_dict["pinky_2"],
                    hand_joint_dict["pinky_0"] * pinky_splay + hand_joint_dict["pinky_1"] * (1 - pinky_splay),
                    (100 - hand_joint_dict["pinky_0"]) * pinky_splay + hand_joint_dict["pinky_1"] * (1 - pinky_splay),
                    ]

                hand_motor_values = np.array(hand_motor_values)
                hand_motor_values = np.clip(hand_motor_values, 0, 100)
                robot.hand_bus.write("Acceleration", 255, hand_motor_names)
                robot.hand_bus.write("Goal_Position", hand_motor_values, hand_motor_names)

                # if i%20==0 and i > 100:
                #     try:
                #         loads = robot.hand_bus.read("Present_Load")
                #         for i, comp in enumerate(hand_components):
                #             # Wave oscillates between 0 and 2024:
                #             # Center (1012) +/- 1012 * sin(...)
                #             comp["value"] = loads[i]
                #     except:
                #         pass


            time.sleep(0.01)

            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         robot.hand_bus.disconnect()
            #         robot.arm_bus.disconnect()
            #         exit()
            #     # Check for user pressing 'q' to quit
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_q:
            #             robot.hand_bus.disconnect()
            #             robot.arm_bus.disconnect()
            #             exit()

            # # Draw background
            # screen.fill((0, 0, 0))  # Black background
            
            # # Draw hand components
            # for comp in hand_components:
            #     x, y = comp["pos"]
            #     color = value_to_color(comp["value"])
            #     pygame.draw.rect(screen, color, (x, y, 30, 30))

            # pygame.display.flip()


if __name__ == "__main__":
    typer.run(main)
