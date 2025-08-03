#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Abstract depth camera interface extending the base Camera class.

Provides depth-specific methods while maintaining full compatibility with regular cameras.
"""

import abc
from typing import Tuple

import numpy as np

from .camera import Camera


class DepthCamera(Camera):
    """Abstract base class for cameras with depth capabilities.
    
    Extends Camera interface with depth-specific methods while maintaining
    full compatibility with base Camera interface.
    
    Design:
    - All DepthCamera instances work as regular Camera instances
    - Depth methods are additive, not replacement
    - Unified RGB+depth read for performance
    - Raw depth data for proper visualization
    """
    
    @abc.abstractmethod
    def read_depth(self) -> np.ndarray:
        """Read raw depth frame synchronously.
        
        Returns:
            np.ndarray: Raw depth as uint16 millimeters, shape (H, W).
                       Values of 0 indicate invalid/no depth measurement.
        """
        pass
    
    
    @abc.abstractmethod
    def async_read_rgb_and_depth(self, timeout_ms: float = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Atomic read of both RGB and depth streams for reliable performance.
        
        This is the primary method for depth cameras. Gets both streams in one
        atomic operation from the camera's background thread, avoiding event
        conflicts and ensuring consistent frame timing.
        
        Args:
            timeout_ms: Maximum wait time in milliseconds.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (rgb_image, raw_depth)
                - rgb_image: RGB as uint8, shape (H, W, 3) 
                - raw_depth: Depth as uint16 millimeters, shape (H, W)
        """
        pass
    
    def has_depth(self) -> bool:
        """Check if camera has depth capability.
        
        Returns:
            bool: True for DepthCamera instances.
        """
        return True
        
    @property
    @abc.abstractmethod
    def depth_scale(self) -> float:
        """Depth scale factor.
        
        Returns:
            float: Scale factor (typically 1.0 for millimeters, 0.001 for meters).
        """
        pass
    
    @property  
    @abc.abstractmethod
    def is_depth_aligned(self) -> bool:
        """Check if depth is aligned to color frame.
        
        Returns:
            bool: True if depth and color frames are pixel-aligned.
        """
        pass