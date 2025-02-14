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
import warnings
from pathlib import Path
from typing import TypeVar

import imageio

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


def write_video(video_path, stacked_frames, fps):
    # Filter out DeprecationWarnings raised from pkg_resources
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
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
                    f"Dictionary keys do not match.\n" f"Expected: {target.keys()}, got: {source.keys()}"
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
