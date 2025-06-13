from lerobot import available_robots


class BiArmSO101:
    
    def __init__(self):
        self.left_robot = available_robots
        self.right_robot = available_robots.so101_leader()