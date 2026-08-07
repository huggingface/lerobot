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

"""Shared dataset recording configuration used by both ``lerobot-record`` and ``lerobot-rollout``."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .video import DepthEncoderConfig, RGBEncoderConfig, depth_encoder_defaults, rgb_encoder_defaults


@dataclass
class DatasetRecordConfig:
    """Shared dataset recording configuration used by both `lerobot-record` and `lerobot-rollout`.

    Args:
        repo_id (`str`, *optional*, defaults to `""`): Dataset identifier. By convention it should match
            `'{hf_username}/{dataset_name}'` (e.g. `lerobot/test`).
        single_task (`str`, *optional*, defaults to `""`): A short but accurate description of the task performed during the
            recording (e.g. `"Pick the Lego block and drop it in the box on the right."`).
        root (`str | Path | None`, *optional*): Root directory where the dataset will be stored (e.g.
            `'dataset/path'`). If `None`, defaults to `$HF_LEROBOT_HOME/repo_id`.
        fps (`int`, *optional*, defaults to 30): Limit the frames per second.
        episode_time_s (`int | float`, *optional*, defaults to 60): Number of seconds for data recording
            for each episode.
        reset_time_s (`int | float`, *optional*, defaults to 60): Number of seconds for resetting the
            environment after each episode.
        num_episodes (`int`, *optional*, defaults to 50): Number of episodes to record.
        video (`bool`, *optional*, defaults to `True`): Encode frames in the dataset into video.
        push_to_hub (`bool`, *optional*, defaults to `True`): Upload dataset to the Hugging Face Hub.
        private (`bool | None`, *optional*): If `True`, upload as private; if `None`, defer to the org
            default on the Hub (only affects orgs).
        tags (`list[str] | None`, *optional*): Add tags to your dataset on the Hub.
        num_image_writer_processes (`int`, *optional*, defaults to 0): Number of subprocesses handling the
            saving of frames as PNG. Set to 0 to use threads only; set to >=1 to use subprocesses, each
            using threads to write images. The best number of processes and threads depends on your
            system. We recommend 4 threads per camera with 0 processes. If fps is unstable, adjust the
            thread count. If still unstable, try using 1 or more subprocesses.
        num_image_writer_threads_per_camera (`int`, *optional*, defaults to 4): Number of threads writing
            the frames as png images on disk, per camera. Too many threads might cause unstable
            teleoperation fps due to the main thread being blocked. Not enough threads might cause low
            camera fps.
        video_encoding_batch_size (`int`, *optional*, defaults to 1): Number of episodes to record before
            batch encoding videos. Set to 1 for immediate encoding (default behavior), or higher for
            batched encoding.
        rgb_encoder (`RGBEncoderConfig`, *optional*): Video encoder settings for camera MP4s (codec,
            quality, GOP, etc.). Tuned via CLI nested keys, e.g. `--dataset.rgb_encoder.vcodec=h264`.
        depth_encoder (`DepthEncoderConfig`, *optional*): Video encoder settings for depth-map MP4s (codec,
            quality, GOP, etc.). Tuned via CLI nested keys.
        streaming_encoding (`bool`, *optional*, defaults to `False`): Enable streaming video encoding:
            encode frames in real-time during capture instead of writing PNG images first. Makes
            `save_episode()` near-instant. More info in the documentation:
            https://huggingface.co/docs/lerobot/streaming_video_encoding
        encoder_queue_maxsize (`int`, *optional*, defaults to 30): Maximum number of frames to buffer per
            camera when using streaming encoding. ~1s buffer at 30fps. Provides backpressure if the encoder
            can't keep up.
        encoder_threads (`int | None`, *optional*): Number of threads per encoder instance. `None` means
            auto (codec default). Lower values reduce CPU usage; maps to `'lp'` (via `svtav1-params`) for
            libsvtav1 and `'threads'` for h264/hevc.
        no_stamp (`bool`, *optional*, defaults to `False`): Skip appending the date-time tag to `repo_id`,
            keeping the user-provided name as-is (e.g. self-managed versioned names intended for a later
            `lerobot-edit-dataset merge`).
    """

    repo_id: str = ""
    single_task: str = ""
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool | None = None
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rgb_encoder: RGBEncoderConfig = field(default_factory=rgb_encoder_defaults)
    depth_encoder: DepthEncoderConfig = field(default_factory=depth_encoder_defaults)
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int | None = None
    no_stamp: bool = False

    def stamp_repo_id(self) -> None:
        """Append a date-time tag to ``repo_id`` so each recording session gets a unique name.

        Must be called explicitly at dataset *creation* time — not on resume,
        where the existing ``repo_id`` (already stamped) must be preserved.
        No-op when ``no_stamp`` is set, preserving a user-managed ``repo_id``.
        """
        if self.no_stamp:
            return
        if self.repo_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.repo_id = f"{self.repo_id}_{timestamp}"
