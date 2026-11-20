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

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.inference import FeatureSpec, ObservationSnapshot
from lerobot.remote_inference.client import parse_capabilities
from lerobot.remote_inference.codec import decode_message, encode_message
from lerobot.remote_inference.protocol import Envelope, MessageType
from tests.inference.policy_helpers import observation
from tests.remote_inference.helpers import admit_session, client_config, client_for


@pytest.mark.parametrize("padded", [False, True])
def test_history_images_and_padding_reach_the_remote_runner(chunk_worker, padded):
    worker = chunk_worker
    key = "observation.images.cam_high"
    camera = FeatureSpec(key, (2, 2, 3), "uint8", "rgb", semantics="RGB")
    worker.runner.capabilities = replace(
        worker.runner.capabilities,
        features=(*worker.runner.capabilities.features, camera),
        current_observation_only=False,
        memory_image_key=key,
        memory_frames=1,
        memory_stride=25,
    )
    worker.runner.predict = Mock(wraps=worker.runner.predict)
    client = client_for(worker, client_config())
    session = admit_session(worker)
    source = observation()
    current = np.full(camera.shape, 200, dtype=np.uint8)
    past = np.full(camera.shape, 75, dtype=np.uint8)
    snapshot = ObservationSnapshot(
        {**source.features, key: current},
        source.capture_time,
        source.task,
        observation_id=source.observation_id,
        history_images=(past,),
        history_is_pad=(padded,),
    )
    request = Envelope(
        MessageType.OBSERVATION,
        worker.instance_id,
        session,
        0,
        "memory-request",
        {**client._observation_body(snapshot), "delay": 0},
    )
    response = worker.submit(decode_message(encode_message(request))).result(2)
    assert response.message_type is MessageType.ACTION, response.body
    restored = worker.runner.predict.call_args.args[0]
    np.testing.assert_array_equal(restored.history_images[0], past)
    assert restored.history_is_pad == (padded,)
    batch = worker.runner._batch(restored)
    assert batch[key].shape == (1, 2, 3, 2, 2)
    assert batch[f"{key}_is_pad"].tolist() == [[padded, False]]


def test_non_memory_wire_schema_remains_compatible_with_old_clients(chunk_worker):
    descriptor = chunk_worker.descriptor
    assert not {"memory_image_key", "memory_frames", "memory_stride"} & descriptor["capabilities"].keys()
    assert parse_capabilities(descriptor["capabilities"]) == chunk_worker.runner.capabilities
    client = client_for(chunk_worker, client_config())
    body = client._observation_body(observation())
    assert "history_images" not in body and "history_is_pad" not in body
    session = admit_session(chunk_worker)
    request = Envelope(
        MessageType.OBSERVATION, chunk_worker.instance_id, session, 0, "legacy-request", {**body, "delay": 0}
    )
    response = chunk_worker.submit(decode_message(encode_message(request))).result(2)
    assert response.message_type is MessageType.ACTION, response.body
