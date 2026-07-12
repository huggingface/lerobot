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


def test_rtc_policy_kwargs_compatibility():
    from lerobot.rollout.inference.rtc import supports_rtc_inference_kwargs

    class FixedSignaturePolicy:
        def predict_action_chunk(self, batch):
            return batch

    class ExplicitRtcPolicy:
        def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None):
            return batch

    class VariadicPolicy:
        def predict_action_chunk(self, batch, **kwargs):
            return batch

    assert not supports_rtc_inference_kwargs(FixedSignaturePolicy())
    assert supports_rtc_inference_kwargs(ExplicitRtcPolicy())
    assert supports_rtc_inference_kwargs(VariadicPolicy())
