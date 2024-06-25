#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import warnings

import imageio


def write_video(video_path, stacked_frames, fps):
    # Filter out DeprecationWarnings raised from pkg_resources
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)


import serial
import os
import time

def reset_usb_port(port):
    try:
        # Close the serial port if it's open
        ser = serial.Serial(port)
        ser.close()
    except serial.serialutil.SerialException as e:
        print(f"Exception while closing the port: {e}")

    # Find the USB device path
    usb_device_path = None
    for root, dirs, files in os.walk('/sys/bus/usb/drivers/usb'):
        for dir_name in dirs:
            if port in dir_name:
                usb_device_path = os.path.join(root, dir_name)
                break

    if usb_device_path:
        # Unbind and rebind the USB device
        try:
            unbind_path = os.path.join(usb_device_path, 'unbind')
            bind_path = os.path.join(usb_device_path, 'bind')
            usb_id = os.path.basename(usb_device_path)

            with open(unbind_path, 'w') as f:
                f.write(usb_id)
            time.sleep(1)  # Wait for a second
            with open(bind_path, 'w') as f:
                f.write(usb_id)
            print(f"USB port {port} has been reset.")
        except Exception as e:
            print(f"Exception during USB reset: {e}")
    else:
        print(f"Could not find USB device path for port: {port}")