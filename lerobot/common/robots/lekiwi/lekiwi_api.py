def main():
    teleop_arm_config = SO100TeleopConfig(port="/dev/tty.usbmodem585A0085511")
    teleop_arm = SO100Teleop(teleop_arm_config)

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)

    robot_config = kiwiconfig(port="/dev/tty.usbmodem575E0032081")
    robot = KiwiRobotDaemon(robot_config)

    teleop_arm.connect()
    keyboard.connect()
    robot.connect() # Establish ZMQ sockets with the mobile robot

    start = time.perf_counter()
    duration = 0
    while duration < 20:
        
        arm_action = teleop_arm.get_action()
        base_action = keyboard.get_action()
        action = {
            **arm_action,
            # **base_action ??
        }
        robot.send_action(action) # sends over ZMQ
        # robot.get_observation() # receives over ZMQ

        print(action)
        duration = time.perf_counter() - start

    robot.disconnect() # cleans ZMQ comms
    teleop.disconnect()

if __name__ == "__main__":
    main()