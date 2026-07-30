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

from threading import Event
from types import SimpleNamespace

import pytest


def test_rtc_policy_must_declare_semantic_support():
    from lerobot.rollout.inference.rtc import supports_rtc_inference

    class VariadicPolicy:
        def supports_rtc(self):
            return False

        def predict_action_chunk(self, batch, **kwargs):
            return batch

    assert not supports_rtc_inference(VariadicPolicy())


def test_rtc_policy_accepts_explicit_or_variadic_kwargs():
    from lerobot.rollout.inference.rtc import supports_rtc_inference

    class ExplicitRtcPolicy:
        def supports_rtc(self):
            return True

        def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None):
            return batch

    class VariadicPolicy:
        def supports_rtc(self):
            return True

        def predict_action_chunk(self, batch, **kwargs):
            return batch

    assert supports_rtc_inference(ExplicitRtcPolicy())
    assert supports_rtc_inference(VariadicPolicy())


@pytest.mark.parametrize("positional_only", [False, True])
def test_rtc_policy_rejects_incompatible_call_shape(positional_only):
    from lerobot.rollout.inference.rtc import supports_rtc_inference

    class FixedSignaturePolicy:
        def supports_rtc(self):
            return True

        if positional_only:

            def predict_action_chunk(self, batch, inference_delay, prev_chunk_left_over, /):
                return batch

        else:

            def predict_action_chunk(self, batch):
                return batch

    assert not supports_rtc_inference(FixedSignaturePolicy())


def test_rtc_policy_rejected_before_robot_creation(monkeypatch):
    import lerobot.rollout.context as rollout_context
    from lerobot.rollout.inference import RTCInferenceConfig

    class IncompatiblePolicy:
        config = SimpleNamespace()

        def supports_rtc(self):
            return False

        def predict_action_chunk(self, batch, **kwargs):
            return batch

    cfg = SimpleNamespace(
        inference=RTCInferenceConfig(),
        policy=SimpleNamespace(type="incompatible", pretrained_path="unused"),
        use_torch_compile=False,
        device="cpu",
    )
    monkeypatch.setattr(rollout_context, "_load_pretrained_policy", lambda _: IncompatiblePolicy())
    monkeypatch.setattr(
        rollout_context,
        "make_robot_from_config",
        lambda _: pytest.fail("robot must not be constructed for an incompatible RTC policy"),
    )

    with pytest.raises(ValueError, match="Use '--inference.type=sync' instead"):
        rollout_context.build_rollout_context(cfg, Event())
