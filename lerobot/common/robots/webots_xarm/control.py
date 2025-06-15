from controller import Supervisor
import subprocess
import os
import time
import numpy as np
from PIL import Image
import random
import time

def open_webots():
    WEBOTS_PATH = "/usr/local/webots/webots"
    WORLD_PATH = "/home/marija/Documents/lerobot/lerobot/common/robots/lite6/data/worlds/lite6_demo.wbt"

    if not os.path.isfile(WEBOTS_PATH):
        print("Cannot find Webots on path :")
        print(WORLD_PATH)
        exit(1)

    if not os.path.isfile(WORLD_PATH):
        print("Cannot find Webots world file on path :")
        print(WORLD_PATH)
        exit(1)

    webots_process = subprocess.Popen([WEBOTS_PATH, WORLD_PATH])
    print("Simulation started!")

    time.sleep(5)
    
class Lite6():
    def __init__(self):
        # open_webots()
        self.robot_ = Supervisor()
        self.timestep_ = int(self.robot_.getBasicTimeStep())
        self.object_ = self.robot_.getFromDef("TARGET")
        self.object_translation_ = self.object_.getField("translation")
        self.object_rotation_ = self.object_.getField("rotation")

        self.joint_names_ = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.motors_ = [self.robot_.getDevice(name) for name in self.joint_names_]

        for motor in self.motors_:
            motor.setPosition(0.0)
    
        self.camera_ = self.robot_.getDevice("camera")
        self.camera_.enable(self.timestep_)
        
        self.left_motor_ = self.robot_.getDevice("gripper_left_joint")
        self.right_motor_ = self.robot_.getDevice("gripper_right_joint")

    
    def get_image(self):
        image = self.camera_.getImage()

        width = self.camera_.getWidth()
        height = self.camera_.getHeight()
        img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        pil_image = Image.fromarray(img_array[:, :, :3], 'RGB')
        
        return pil_image
    
    def set_position(self, goal):
        for motor, target in zip(self.motors_, goal):
            motor.setPosition(target)
    
    def randomize_object(self):
        pos = list(self.object_translation_.getSFVec3f())
        pos[0] += 0.02
        pos[1] += 0.01
        self.object_translation_.setSFVec3f(pos)
        
        orientation = list(self.object_rotation_.getSFVec3f())
        orientation[0] += 0.2
        self.object_translation_.setSFVec3f(orientation)
        print(orientation)
        print(pos)
    
    def step(self):
        return self.robot_.step(self.timestep_) != -1
    
    def get_timestap(self):
        return self.timestep_
            
        
if __name__ == "__main__":
    robot = Lite6()
    goal_positions = [
        [0.0, -0.5, 0.7, -1.2, 0.4, 0.0],  
        [0.3, -0.3, 0.6, -1.0, 0.3, 0.2],
        [-0.3, -0.7, 0.8, -1.3, 0.5, -0.2],
        [0.2, -0.6, 0.5, -1.1, 0.6, 0.1],
        [-0.1, -0.4, 0.7, -0.9, 0.4, 0.0]
    ]
    
    goal = [0.5, -0.3, 0.8, -1.0, 0.2, 0.5]
    elapsed_time = 0
    interval = 500
    
    while robot.step():
        robot.set_position(goal)
        image = robot.get_image()
        elapsed_time += robot.get_timestap()
        
        if elapsed_time >= interval:
            goal = random.choice(goal_positions)
            elapsed_time = 0
            robot.randomize_object()