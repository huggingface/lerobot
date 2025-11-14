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

from dataclasses import dataclass #we are using this because this is a data holder class (since its a config file) where its main operation is just to store data, very minimal edits

from ..config import TeleoperatorConfig #goes back one folder to import a base config file that defines how any teleoperator configuration should behave

@TeleoperatorConfig.register_subclass("earthrover_keyboard")
@dataclass
class EarthroverKeyboardTeleopConfig(TeleoperatorConfig):
    #check if we want to state what keys we will capture/listen to
    port: int = 8888 #Port to connect to the earthrover
    ip: str = "192.168.11.1 "#Robot's IP to connect to the earthrover

@TeleoperatorConfig.register_subclass("earthrover_keyboard_actions")
@dataclass
class EarthroverKeyboardTeleopConfigActions(EarthroverKeyboardTeleopConfig):
#     this class specifically controls the end effector, commented out for now
    #use_gripper: bool = True
    use_key: bool = True #placeholder for some random state we need to decide