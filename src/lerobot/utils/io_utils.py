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
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]


class StreamingVideoWriter:
    """Incrementally encode RGB frames to an MP4 without retaining them in memory."""

    def __init__(self, video_path: str | Path, fps: int) -> None:
        from .import_utils import require_package

        require_package("av", extra="av-dep")
        import av

        self._av = av
        self._container = av.open(str(video_path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._shape: tuple[int, int] | None = None
        self.frames_written = 0

    def add_frame(self, frame_array) -> None:
        orig_height, orig_width = frame_array.shape[:2]
        height = orig_height - orig_height % 2
        width = orig_width - orig_width % 2
        if self._shape is None:
            self._shape = (height, width)
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = "yuv420p"
        elif self._shape != (height, width):
            raise ValueError(f"Video frame shape changed from {self._shape} to {(height, width)}")
        frame = self._av.VideoFrame.from_ndarray(frame_array[:height, :width], format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames_written += 1

    def close(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None


def load_json(fpath: Path) -> Any:
    """Load data from a JSON file.

    Args:
        fpath (Path): Path to the JSON file.

    Returns:
        Any: The data loaded from the JSON file.
    """
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_video(video_path: str | Path, stacked_frames: list, fps: int) -> None:
    """Write a sequence of RGB frames to an MP4 video file using libx264.

    Args:
        video_path: Output file path.
        stacked_frames: List of HWC uint8 numpy arrays (RGB).
        fps: Frames per second for the output video.
    """
    writer = StreamingVideoWriter(video_path, fps)
    try:
        for frame_array in stacked_frames:
            writer.add_frame(frame_array)
    finally:
        writer.close()


def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:
    """
    Loads the JSON data from `fpath` and recursively fills `obj` with the
    corresponding values (strictly matching structure and types).
    Tuples in `obj` are expected to be lists in the JSON data, which will be
    converted back into tuples.
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        """
        Recursively overwrite the structure in `target` with data from `source`,
        performing strict checks on structure and type.
        Returns the updated version of `target` (especially important for tuples).
        """

        # If the target is a dictionary, source must be a dictionary as well.
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")

            # Check that they have exactly the same set of keys.
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )

            # Recursively update each key.
            for k in target:
                target[k] = _deserialize(target[k], source[k])

            return target

        # If the target is a list, source must be a list as well.
        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")

            # Check length
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")

            # Recursively update each element.
            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])

            return target

        # If the target is a tuple, the source must be a list in JSON,
        # which we'll convert back to a tuple.
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")

            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")

            # Convert each element, forming a new tuple.
            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))

            # Return a brand new tuple (tuples are immutable in Python).
            return tuple(converted_items)

        # Otherwise, we're dealing with a "primitive" (int, float, str, bool, None).
        else:
            # Check the exact type.  If these must match 1:1, do:
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    # Perform the in-place/recursive deserialization
    updated_obj = _deserialize(obj, data)
    return updated_obj
