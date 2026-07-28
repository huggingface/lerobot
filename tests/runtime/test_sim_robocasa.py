# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import sys
from types import SimpleNamespace

import numpy as np

from lerobot.runtime.sim_robocasa import RoboCasaSimBackend
from lerobot.utils.video_annotation import annotate_frame


def test_overlay_draws_each_label_once(monkeypatch):
    put_text_calls = []
    rectangle_calls = []

    def put_text(image, text, origin, font, scale, color, thickness, line_type):
        put_text_calls.append((text, color, thickness))
        return image

    def rectangle(image, start, end, color, thickness):
        rectangle_calls.append((start, end, color, thickness))
        return image

    def add_weighted(src1, alpha, src2, beta, gamma, *, dst):
        dst[:] = src1 * alpha + src2 * beta + gamma
        return dst

    fake_cv2 = SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=16,
        getTextSize=lambda text, font, scale, thickness: ((len(text) * 7, 10), 0),
        putText=put_text,
        rectangle=rectangle,
        addWeighted=add_weighted,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    frame = np.full((120, 480, 3), 200, dtype=np.uint8)
    annotated = annotate_frame(
        frame,
        (("Task", "close the fridge"), ("Subtask", "reach for the handle"), ("Memory", None)),
    )

    assert [call[0] for call in put_text_calls] == [
        "Task: close the fridge",
        "Subtask: reach for the handle",
    ]
    assert all(color == (255, 255, 255) and thickness == 1 for _, color, thickness in put_text_calls)
    assert len(rectangle_calls) == 1
    assert not np.shares_memory(annotated, frame)


def test_capture_updates_live_frame_when_recording_is_disabled(monkeypatch):
    backend = object.__new__(RoboCasaSimBackend)
    frame = np.full((8, 8, 3), 42, dtype=np.uint8)
    written = []
    backend.record = False
    backend.runtime_state = None
    backend._multiview_frame = lambda: frame
    backend._current_task = lambda: "task"
    backend._subtask_getter = None
    backend._memory_getter = None
    backend._latest_frame = None
    backend._write_live_frame = written.append
    monkeypatch.setattr("lerobot.runtime.sim_robocasa.annotate_frame", lambda image, labels: image)

    backend._capture_frame()

    assert backend._latest_frame is frame
    assert written == [frame]
