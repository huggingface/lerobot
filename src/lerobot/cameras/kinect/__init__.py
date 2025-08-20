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

import os

from .camera_kinect import KinectCamera
from .configuration_kinect import KinectCameraConfig, KinectPipeline


def available_kinect():
    """Check if Kinect v2 support is available.

    Returns True if pylibfreenect2 can be imported and libfreenect2 is properly installed.
    Provides detailed error messages to help with troubleshooting.
    """
    try:
        import pylibfreenect2

        return True
    except ImportError as e:
        import logging

        logger = logging.getLogger(__name__)

        # Check for libfreenect2 installation
        libfreenect2_prefix = os.environ.get("LIBFREENECT2_INSTALL_PREFIX")

        error_msg = (
            "\n" + "=" * 60 + "\n"
            "Kinect v2 camera support is not available.\n\n"
            "To use Kinect v2 cameras, you need to:\n\n"
            "1. Install libfreenect2 library first:\n"
            "   - Windows: Follow https://github.com/OpenKinect/libfreenect2#windows\n"
            "   - Linux: Follow https://github.com/OpenKinect/libfreenect2#linux\n"
            "   - macOS: Follow https://github.com/OpenKinect/libfreenect2#macos\n\n"
            "2. Set the LIBFREENECT2_INSTALL_PREFIX environment variable:\n"
            f"   Current value: {libfreenect2_prefix or 'NOT SET'}\n"
            "   Example: export LIBFREENECT2_INSTALL_PREFIX=/path/to/libfreenect2\n\n"
            "3. Install the Python bindings:\n"
            "   pip install 'lerobot[kinect]'\n"
            "   or\n"
            "   pip install git+https://github.com/cerealkiller2527/pylibfreenect2-py310.git\n\n"
            "Note: Python 3.10+ is required for Kinect v2 support.\n"
            f"Import error: {e}\n" + "=" * 60
        )
        logger.debug(error_msg)
        return False


__all__ = ["KinectCamera", "KinectCameraConfig", "KinectPipeline", "available_kinect"]
