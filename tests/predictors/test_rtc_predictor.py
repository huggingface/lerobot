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

"""Unit tests for the RTC engine's overhead time-advance step.

These exercise ``RTCInferenceEngine._time_advanced_obs`` in isolation (no policy
or robot) with a scripted predictor, so the obs-shift logic is tested
independently of the cube detector.
"""

import numpy as np

from lerobot.predictors import PredictorOutput
from lerobot.rollout.inference.rtc import RTCInferenceEngine


def _frame_with_red_square(x0: int, y0: int = 20, size: int = 10) -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[y0 : y0 + size, x0 : x0 + size, 0] = 255
    return frame


class _ScriptedPredictor:
    """Returns a fixed ``PredictorOutput`` and the frame's red mask."""

    def __init__(self, output: PredictorOutput):
        self.output = output
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return self.output

    def red_mask(self, frame):
        return frame[..., 0] > 0


def _engine_with_predictor(predictor, camera="overall") -> RTCInferenceEngine:
    eng = object.__new__(RTCInferenceEngine)
    eng._predictor = predictor
    eng._predictor_camera = camera
    eng._predictor_mode = "image_shift"
    return eng


def _engine_with_engage_gate(predictor, *, threshold=0.5, direction="positive"):
    eng = _engine_with_predictor(predictor)
    eng._predictor_mode = "engage_gate"
    eng._engaged = False
    eng._last_engage_frame_id = None
    eng._engage_axis = "x"
    eng._engage_threshold = threshold
    eng._engage_direction = direction
    eng._engage_lead_s = 0.5
    return eng


def test_time_advanced_obs_shifts_cube_by_velocity_times_lead():
    pred = _ScriptedPredictor(
        PredictorOutput(target_visible=True, center_px=(14.5, 24.5), velocity_px_s=(200.0, 0.0))
    )
    eng = _engine_with_predictor(pred)
    frame = _frame_with_red_square(10)
    obs = {"overall": frame, "observation.state": np.zeros(6)}

    # delay=2 steps * 0.05 s/step = 0.1 s lead -> +20px in x.
    advanced = eng._time_advanced_obs(obs, delay=2, time_per_chunk=0.05)

    assert advanced is not obs  # shallow copy, holder untouched
    assert advanced["observation.state"] is obs["observation.state"]  # other keys preserved
    shifted = advanced["overall"]
    assert shifted[20:30, 10:20, 0].max() == 0  # erased from original spot
    assert shifted[20:30, 30:40, 0].min() == 255  # re-pasted +20px right
    assert np.array_equal(obs["overall"], frame)  # original frame not mutated


def test_time_advanced_obs_noop_when_delay_zero():
    pred = _ScriptedPredictor(PredictorOutput(center_px=(1.0, 2.0), velocity_px_s=(200.0, 0.0)))
    eng = _engine_with_predictor(pred)
    obs = {"overall": _frame_with_red_square(10)}
    assert eng._time_advanced_obs(obs, delay=0, time_per_chunk=0.05) is obs
    assert pred.calls == 0


def test_time_advanced_obs_noop_without_velocity():
    pred = _ScriptedPredictor(PredictorOutput(target_visible=True, center_px=(1.0, 2.0)))
    eng = _engine_with_predictor(pred)
    obs = {"overall": _frame_with_red_square(10)}
    assert eng._time_advanced_obs(obs, delay=2, time_per_chunk=0.05) is obs


def test_time_advanced_obs_noop_when_disabled_or_camera_missing():
    obs = {"overall": _frame_with_red_square(10)}
    disabled = _engine_with_predictor(None)
    assert disabled._time_advanced_obs(obs, delay=2, time_per_chunk=0.05) is obs

    pred = _ScriptedPredictor(PredictorOutput(center_px=(1.0, 2.0), velocity_px_s=(1.0, 0.0)))
    eng = _engine_with_predictor(pred, camera="missing")
    assert eng._time_advanced_obs(obs, delay=2, time_per_chunk=0.05) is obs


def test_engage_gate_opens_from_predicted_center_and_latches():
    pred = _ScriptedPredictor(
        PredictorOutput(target_visible=True, center_px=(20.0, 24.0), velocity_px_s=(30.0, 0.0))
    )
    eng = _engine_with_engage_gate(pred, threshold=0.5)
    first = _frame_with_red_square(10)

    # Predicted x = 20 + 30 * 0.5 = 35; 35 / 63 > 0.5.
    assert eng._evaluate_engage_gate({"overall": first}) is True
    assert eng._engaged is True
    # Once opened, the gate stays open even if the target later disappears.
    pred.output = PredictorOutput(target_visible=False)
    assert eng._evaluate_engage_gate({"overall": np.zeros_like(first)}) is True


def test_engage_gate_waits_when_target_is_missing_or_before_line():
    pred = _ScriptedPredictor(PredictorOutput(target_visible=False))
    eng = _engine_with_engage_gate(pred, threshold=0.8)
    assert eng._evaluate_engage_gate({"overall": _frame_with_red_square(10)}) is False

    pred.output = PredictorOutput(target_visible=True, center_px=(20.0, 24.0))
    assert eng._evaluate_engage_gate({"overall": _frame_with_red_square(11)}) is False
