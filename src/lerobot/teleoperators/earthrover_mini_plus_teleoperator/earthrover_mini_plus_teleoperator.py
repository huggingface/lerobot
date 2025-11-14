#!/usr/bin/env python

# Copyright 2025 SIGRobotics team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_earthrover_mini_plus_teleoperator import EarthroverKeyboardTeleopConfig, EarthroverKeyboardTeleopConfigActions
#from .earthrover_mini_plus_teleoperator import EarthroverMiniPlus, EarthroverKeyboardTeleop TODO: Check if I need this

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

#TODO: Figure out what is Teleoperator
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents #TODO: Figure out if I need this

PYNPUT_AVAILABLE = True #this is just a flag to see whether PYNPUT is able to be imported or not
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally. This is because pynput utilizes a GUI and your OS global input system to function. However, you do not have a GUI (and thus are running a headless linux system).")
    
    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e: #catches any other errors and displays them
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")

class EarthroverKeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """
    
    config_class = EarthroverKeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: EarthroverKeyboardTeleopConfig): #prepares everything an object needs (runs automatically) such as the variables needed, config tells what kind of robot this is
        super().__init__(config) #this tells the parent class to do its setup first (calls the parent class's __init__ function [its constructor])
        self.config = config #saves the setting box into the object
        self.robot_type = config.type #saves the setting box type (what kind of robot) into the object

        self.event_queue = Queue() #creates a queue for key presses
        self.current_pressed = {} #creates a dictionary to track which keys are pressed down at any point in time
        self.listener = None #prepares what will actually listen for keyboard inputs
        self.logs = {} #sets up a dictionary to log all key presses

    @property #turns this function into a read-only function like a variable
    def action_features(self) -> dict:  #describing all properties of an action in an array; creating a blueprint/metadata
        return {
            "dtype": "float32", #data type of values in the action, like how motor positions are this data type
            "shape": (3,), #four arguments to pass in to move the robot (size of the array/how many values)
            "names": { #describing what each element represents
                "fields": ["duration", "speed", "angular"] #TODO: check if i need to set them to have default values later on
            },
        }
    
    @property
    def feedback_features(self) -> dict: #describes the shape of the feedback getting sent back
        return{
            "RPM": {"dtype": "float32", "shape": (4,)}, #the (4,d) represents a 1-D array using a tuple
            "Head": {"dtype": "float32", "shape": (1,)},
            "V": {"dtype": "float32", "shape": (1,)},
            "I": {"dtype": "float32", "shape": (1,)},
            "ACC": {"dtype": "float32", "shape": (3,)}, #acceleration
            "Gyro": {"dtype": "float32", "shape": (3,)},
            "Mag": {"dtype": "float32", "shape": (3,)}
        }
    
    @property #TODO: Create issue because in orig. code LeRobot code is unsafe
    def is_connected(self) -> bool: #self represents the current object calling this method
        if not PYNPUT_AVAILABLE or keyboard is None: #returns whether the robot can be connected or not
            return False
        return isinstance(self.listener, keyboard.Listener) and self.listener.is_alive() #returns true if robot is able to be connected and is
    
    @property #TODO: Check where this is being called and see if we can set this to be true/false
    def is_calibrated(self) -> bool: #should I suppress this function for now?
        pass
    
    #TODO: check how this will be called becuse of the passed in parameter
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "The earthrover-keyboard teleoperation is already setup. Do not run `robot.connect()` twice."
            )
        
        if PYNPUT_AVAILABLE: #runs if robot is not connected yet
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener( #erroring because there's chance PYNPUT gets skipped
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start() #runs in a different thread
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None
    
    #TODO: Check where this is being called and if this needs to be implemented/how
    def calibrate(self) -> None:
        # do i do this: return super().calibrate() or is there a different thing to do
        pass


    def _on_press(self, key): #key is the key being pressed in, and puts what the character is pressed in the queue
        if hasattr(key, "char"):
            self.event_queue.put((key.char, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False)) #pushes this to a queue
        if key == keyboard.Key.esc: #our disconnect key
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty(): #runs while queue is not empty
            key_char, is_pressed = self.event_queue.get_nowait() #returns a tuple of each key state
            self.current_pressed[key_char] = is_pressed #adds to the dictionary

    def configure(self): #TODO: set this up
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter() #gets precise time

        if not self.is_connected: #checks that you are connected
            raise DeviceNotConnectedError(
                "Earthrover Mini Plus Keyboard Teleop is not connected. You need to run `connect()` before `get_action()`."
            )
        
        self._drain_pressed_keys() #updates which keys are pressed down

        # Generate action based on current key states (through creating a set of all currently pressed keys)
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t #checks how long the action took for the current key

        return dict.fromkeys(action, None) #sets the value from each action to be None
    
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass #TODO: Implement this

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Earthrover Mini Plus Keyboard Teleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()

class EarthroverKeyboardTeleopActions(EarthroverKeyboardTeleop): #child class extending parent behavior
    """
    Keyboard teleop class to use keyboard inputs for robot actions.
    Designed to be used with the `Earthrover Mini Plus` robot.
    """

    config_class = EarthroverKeyboardTeleopConfigActions
    name = "earthrover keyboard teleop actions"

    def __init__(self, config: EarthroverKeyboardTeleopConfigActions):
        super().__init__(config) #has parent class set up first
        self.config = config #stores the config inside this object 
        self.misc_keys_queue = Queue() #queue for misc. key presses separate from main robot tasks (different thread)

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {
                "fields": ["duration", "speed", "angular"]
            },
        }
    
    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )
        
        self._drain_pressed_keys()
        duration = 0.0 #TODO: have users be able to enter in terminal
        speed = 0.0
        angular = 0.0

        # Generate action based on current key states
        #TODO: See if I need a quit key
        for key, val in self.current_pressed.items(): #all of the below will error
            if key == "up": #TODO: add in gradient stuff + better way for users to increment
                speed += 5.0
                print(speed)
            elif key == "down":
                speed -= 5.0
                print(speed)
            elif key == "+":
                duration += .5
                print(duration)
            elif key == "-":
                duration -= .5
                print(duration)
            elif key == "left":
                angular -= 0.5
                print(angular)
            elif key == "right":
                angular += 0.5
                print(angular)
            elif val:
                #stores any other misc. keys in the queue
                #can use this to implement other actions like the shortcuts for recording episodes or other interventions
                self.misc_keys_queue.put(key)
        
        if not any(self.current_pressed.values()): 
            duration = 50.0
            speed = 5.0
            angular = 10

        self.current_pressed.clear()

        action_dict = {
            "duration": duration,
            "speed": speed,
            "angular": angular,
        }

        return action_dict

# logger = logging.getLogger(__name__)

# class EarthroverMiniPlus(Teleoperator):
#     """
#     Earthrover Mini Plus designed by SIGRobotics and FrodoBots AI.
#     """

#     config_class = EarthroverMiniPlusConfig
#     name = "earthrover_mini_plus"

#     def __init__(self, config: EarthroverMiniPlusConfig):
#         super().__init__(config)
#         self.config = config
        