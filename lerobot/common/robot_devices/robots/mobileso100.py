class MobileSO100:
    def __init__(self, motor_bus):
        """
        Initializes the MobileSO100 with Feetech motors bus.
        """
        self.motor_bus = motor_bus
        self.motor_ids = ["wheel_1", "wheel_2", "wheel_3"]

        # Initialize motors in velocity mode.
        self.motor_bus.write("Lock", 0)
        self.motor_bus.write("Mode", [1, 1, 1], self.motor_ids)
        self.motor_bus.write("Lock", 1)
        print("Motors set to velocity mode.")

    def read_velocity(self):
        """
        Reads the raw speeds for all wheels. Returns a dictionary with motor index strings:
        {
            "1": raw_speed_wheel_1,
            "2": raw_speed_wheel_2,
            "3": raw_speed_wheel_3
        }
        """
        raw_speeds = self.motor_bus.read("Present_Speed", self.motor_ids)
        return {"1": int(raw_speeds[0]), "2": int(raw_speeds[1]), "3": int(raw_speeds[2])}

    def set_velocity(self, command_speeds):
        """
        Sends raw velocity commands (16-bit encoded values) directly to the motor bus.
        The order of speeds must correspond to self.motor_ids.
        """
        self.motor_bus.write("Goal_Speed", command_speeds, self.motor_ids)

    def stop(self):
        """Stops the robot by setting all motor speeds to zero."""
        self.motor_bus.write("Goal_Speed", [0, 0, 0], self.motor_ids)
        print("Motors stopped.")
