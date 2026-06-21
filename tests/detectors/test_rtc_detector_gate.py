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

"""Unit tests for the RTC engine's detector-driven replan gate.

These exercise ``RTCInferenceEngine._evaluate_detector`` in isolation (no policy
or robot) with a scripted detector, so the gate logic is tested independently of
detector timing.
"""

import numpy as np
import pytest

from lerobot.detectors import DetectorOutput
from lerobot.rollout.inference.rtc import RTCInferenceEngine


class _ScriptedDetector:
    """Returns a fixed ``DetectorOutput`` and counts how often it was called."""

    def __init__(self, output: DetectorOutput):
        self.output = output
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return self.output


def _engine_with_detector(detector, *, chunk_size=50, cooldown_s=0.0) -> RTCInferenceEngine:
    eng = object.__new__(RTCInferenceEngine)
    eng._detector = detector
    eng._supervisor_camera = "cam"
    eng._supervisor_cooldown_s = cooldown_s
    eng._target_visible_required = False
    eng._detector_waiting_for_target = False
    eng._chunk_size = chunk_size
    eng._last_detector_fire = -1.0
    eng._last_detector_frame_id = None
    eng._dynamic_queue_threshold = None
    return eng


def test_effective_threshold_maps_fraction_to_absolute():
    det = _ScriptedDetector(DetectorOutput(effective_chunk_size_threshold=0.5))
    eng = _engine_with_detector(det, chunk_size=50)

    replan, threshold = eng._evaluate_detector({"cam": np.zeros((4, 4, 3))})

    assert replan is False
    assert threshold == pytest.approx(25.0)  # 0.5 * chunk_size


def test_same_frame_is_not_re_evaluated():
    det = _ScriptedDetector(DetectorOutput(effective_chunk_size_threshold=0.4))
    eng = _engine_with_detector(det, chunk_size=50)
    frame = np.zeros((4, 4, 3))

    eng._evaluate_detector({"cam": frame})
    _, threshold = eng._evaluate_detector({"cam": frame})  # identical object

    assert det.calls == 1  # detector only ran for the first (new) frame
    assert threshold == pytest.approx(20.0)  # cached threshold reused


def test_urgent_replan_fires_once_then_respects_cooldown():
    det = _ScriptedDetector(DetectorOutput(replan_now=True, reason="red_cube_urgent_speed"))
    eng = _engine_with_detector(det, cooldown_s=100.0)

    first, _ = eng._evaluate_detector({"cam": np.zeros((4, 4, 3))})
    second, _ = eng._evaluate_detector({"cam": np.ones((4, 4, 3))})

    assert first is True
    assert second is False  # suppressed by cooldown


def test_target_visibility_gate_tracks_waiting_state():
    det = _ScriptedDetector(DetectorOutput(target_visible=False, reason="red_cube_not_visible"))
    eng = _engine_with_detector(det)
    eng._target_visible_required = True

    replan, _ = eng._evaluate_detector({"cam": np.zeros((4, 4, 3))})

    assert replan is False
    assert eng._detector_waiting_for_target is True

    det.output = DetectorOutput(target_visible=True, center_px=(1.0, 2.0), reason="red_cube_initialized")
    replan, _ = eng._evaluate_detector({"cam": np.ones((4, 4, 3))})

    assert replan is False
    assert eng._detector_waiting_for_target is False


def test_target_visibility_gate_suppresses_queue_replan_only():
    det = _ScriptedDetector(DetectorOutput())
    eng = _engine_with_detector(det)
    eng._target_visible_required = True
    eng._detector_waiting_for_target = True

    assert eng._should_run_inference(queue_size=0, effective_threshold=30, detector_replan=False) is False
    assert eng._should_run_inference(queue_size=0, effective_threshold=30, detector_replan=True) is True


def test_missing_camera_reuses_cached_threshold():
    det = _ScriptedDetector(DetectorOutput(replan_now=True))
    eng = _engine_with_detector(det)
    eng._dynamic_queue_threshold = 7.0

    replan, threshold = eng._evaluate_detector({"other_cam": np.zeros((4, 4, 3))})

    assert replan is False
    assert threshold == pytest.approx(7.0)
    assert det.calls == 0
