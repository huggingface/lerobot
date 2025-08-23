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


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

class ExperimentLogger(ABC):
    """Abstract base class for experiment logging backends."""
    
    @abstractmethod
    def __init__(self, cfg: Any):
        """Initialize the logger with configuration."""
        pass
    
    @abstractmethod
    def log_dict(
        self, 
        d: Dict[str, Any], 
        step: int | None = None, 
        mode: str = "train", 
        custom_step_key: str | None = None
    ) -> None:
        """Log a dictionary of metrics.
        
        Args:
            d: Dictionary of metrics to log
            step: Global step number
            mode: Logging mode ("train" or "eval")
            custom_step_key: Custom step key for asynchronous training
        """
        pass
    
    @abstractmethod
    def log_policy(self, checkpoint_dir: Path) -> None:
        """Log policy checkpoint as an artifact.
        
        Args:
            checkpoint_dir: Path to the checkpoint directory
        """
        pass
    
    @abstractmethod
    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """Log a video file.
        
        Args:
            video_path: Path to the video file
            step: Global step number
            mode: Logging mode ("train" or "eval")
        """
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Clean up and finish the logging session."""
        pass
