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

"""Golden parity contract test: remote request path == local RTC compute path.

The local side replicates exactly what ``RTCInferenceEngine._rtc_loop``
(rtc.py) does per iteration; the remote side runs the same observation
through the wire codec (encode -> decode), ``PolicyServer.run_inference_request``,
and the action-chunk codec — no network, no threads.  With the same
deterministic policy and identical inputs, both ActionQueues must stay
byte-identical merge after merge.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("msgpack")

from lerobot.policies.rtc import ActionQueue  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.policy_server import codec  # noqa: E402
from lerobot.policy_server.schema import MsgHeader, ObservationMsg  # noqa: E402
from lerobot.policy_server.server import _normalize_prev_actions_length  # noqa: E402
from lerobot.policy_server.session import Session  # noqa: E402
from lerobot.utils.constants import OBS_STATE, OBS_STR  # noqa: E402
from lerobot.utils.feature_utils import build_dataset_frame  # noqa: E402
from tests.policy_server.conftest import (  # noqa: E402
    ACTION_NAMES,
    CHUNK_SIZE,
    STATE_DIM,
    TASK,
    MockChunkPolicy,
    make_logic_server,
    make_mock_processors,
    make_robot_obs,
)

# Must match make_manifest()'s default RTCConfig (enabled=True, horizon=10).
EXECUTION_HORIZON = 10
ROBOT_TYPE = "mock_robot"
# Fixed per-cycle inference-delay hints; cycle 2 exercises a non-zero delay.
DELAYS = [0, 2, 1]
# Actions consumed from both queues between cycles (makes prefixes non-trivial).
CONSUME_K = 4

STATE_ONLY_FEATURES = {
    OBS_STATE: {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": list(ACTION_NAMES),
    },
}


def _make_queue() -> ActionQueue:
    return ActionQueue(RTCConfig(enabled=True, execution_horizon=EXECUTION_HORIZON))


def _local_cycle(policy, pre, post, queue, features, obs, delay) -> None:
    """Replicates the loop body of RTCInferenceEngine._rtc_loop (rtc.py)."""
    idx_before = queue.get_action_index()
    prev_actions = queue.get_left_over()

    obs_batch = build_dataset_frame(features, obs, prefix=OBS_STR)
    obs_batch = prepare_observation_for_inference(obs_batch, torch.device("cpu"), TASK, ROBOT_TYPE)
    obs_batch["task"] = [TASK]
    preprocessed = pre(obs_batch)

    if prev_actions is not None:
        prev_actions = _normalize_prev_actions_length(prev_actions, target_steps=EXECUTION_HORIZON)

    actions = policy.predict_action_chunk(
        preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
    )
    original = actions.squeeze(0).clone()
    processed = post(actions).squeeze(0)
    queue.merge(original, processed, delay, idx_before)


def _remote_cycle(server, session, queue, features, obs, delay, seq_id) -> None:
    """Replicates RemoteInferenceEngine._request_cycle with the wire codec in
    the loop (encode -> decode on both legs) but no network or threads."""
    idx_before = queue.get_action_index()

    obs_frame = build_dataset_frame(features, obs, prefix=OBS_STR)
    state = obs_frame.pop(OBS_STATE, None)
    images = {k: v for k, v in obs_frame.items() if isinstance(v, np.ndarray) and v.ndim == 3}

    prefix_model: np.ndarray | None = None
    prefix_robot: np.ndarray | None = None
    left_over = queue.get_left_over()
    if left_over is not None and left_over.numel():
        prefix_model = left_over[:EXECUTION_HORIZON].to(torch.float32).numpy()
    processed_left_over = queue.get_processed_left_over()
    if processed_left_over is not None and processed_left_over.numel():
        prefix_robot = processed_left_over[:EXECUTION_HORIZON].to(torch.float32).numpy()

    msg = ObservationMsg(
        state=state,
        images=images,
        task=TASK,
        inference_delay_steps=delay,
        prefix_model=prefix_model,
        prefix_robot=prefix_robot,
        jpeg_quality=0,  # raw image codec: byte-exact transport
    )
    decoded = codec.decode_observation(codec.encode_observation(msg))

    # The float32 wire dtype must introduce zero drift: byte-exact roundtrip.
    assert decoded.state.dtype == np.float32
    assert decoded.state.tobytes() == np.ascontiguousarray(state).tobytes()
    if prefix_model is not None:
        assert decoded.prefix_model.tobytes() == np.ascontiguousarray(prefix_model).tobytes()
    if prefix_robot is not None:
        assert decoded.prefix_robot.tobytes() == np.ascontiguousarray(prefix_robot).tobytes()
    for name, img in images.items():
        assert np.array_equal(decoded.images[name], img)

    reply = server.run_inference_request(session, MsgHeader(seq_id=seq_id), decoded)
    chunk = codec.decode_action_chunk(codec.encode_action_chunk(reply))

    # Reply leg is byte-exact too (float32 in, float32 on the wire).
    assert chunk.chunk_model.tobytes() == np.ascontiguousarray(reply.chunk_model).tobytes()
    assert chunk.chunk_robot.tobytes() == np.ascontiguousarray(reply.chunk_robot).tobytes()

    queue.merge(
        torch.from_numpy(np.ascontiguousarray(chunk.chunk_model)),
        torch.from_numpy(np.ascontiguousarray(chunk.chunk_robot)),
        delay,
        idx_before,
    )


def _drive_parity(features) -> tuple[MockChunkPolicy, MockChunkPolicy]:
    """Run DELAYS cycles through both paths, asserting queue parity after
    each merge and consuming CONSUME_K actions from both queues between
    cycles. Returns (local_policy, remote_policy) for call-level checks."""
    policy_local = MockChunkPolicy()
    pre_local, post_local = make_mock_processors()
    queue_local = _make_queue()

    policy_remote = MockChunkPolicy()
    server = make_logic_server(policy=policy_remote)
    pre_remote, post_remote = make_mock_processors()
    session = Session(
        session_id="parity",
        client_uuid="parity-client",
        task=TASK,
        robot_type=ROBOT_TYPE,
        rtc_enabled=True,
        preprocessor=pre_remote,
        postprocessor=post_remote,
    )
    queue_remote = _make_queue()

    for cycle, delay in enumerate(DELAYS):
        obs = make_robot_obs(seed=float(cycle + 1))
        _local_cycle(policy_local, pre_local, post_local, queue_local, features, obs, delay)
        _remote_cycle(server, session, queue_remote, features, obs, delay, seq_id=cycle + 1)

        assert queue_local.queue is not None and queue_remote.queue is not None
        assert torch.equal(queue_local.queue, queue_remote.queue), (
            f"robot-space queues diverged (cycle {cycle})"
        )
        assert torch.equal(queue_local.original_queue, queue_remote.original_queue), (
            f"model-space queues diverged (cycle {cycle})"
        )
        assert queue_local.queue.shape == (CHUNK_SIZE - min(delay, CHUNK_SIZE), len(ACTION_NAMES))

        # Consume the same k actions on both sides so the next cycle's RTC
        # prefixes are non-trivial (and identical).
        for _ in range(CONSUME_K):
            action_local = queue_local.get()
            action_remote = queue_remote.get()
            assert action_local is not None and action_remote is not None
            assert torch.equal(action_local, action_remote)

    return policy_local, policy_remote


def test_remote_path_matches_local_rtc_path_state_only():
    """3 cycles, state-only features: queues stay byte-identical."""
    _drive_parity(STATE_ONLY_FEATURES)


def test_remote_path_matches_local_rtc_path_with_images(hw_features):
    """Images in the loop (raw codec) must not perturb state-driven outputs."""
    _drive_parity(hw_features)


def test_policy_inputs_identical_across_paths(hw_features):
    """The strongest contract: both policies saw byte-identical inputs."""
    policy_local, policy_remote = _drive_parity(hw_features)

    assert len(policy_local.calls) == len(policy_remote.calls) == len(DELAYS)
    for i, (local_call, remote_call) in enumerate(zip(policy_local.calls, policy_remote.calls, strict=True)):
        assert torch.equal(local_call["state"], remote_call["state"]), f"state diverged (call {i})"
        assert local_call["state"].dtype == remote_call["state"].dtype == torch.float32
        assert local_call["inference_delay"] == remote_call["inference_delay"] == DELAYS[i]
        if local_call["prev_chunk_left_over"] is None:
            assert remote_call["prev_chunk_left_over"] is None
        else:
            assert torch.equal(local_call["prev_chunk_left_over"], remote_call["prev_chunk_left_over"])
            assert local_call["prev_chunk_left_over"].shape == (EXECUTION_HORIZON, len(ACTION_NAMES))

    # First cycle has no leftover; later cycles must carry a real prefix.
    assert policy_local.calls[0]["prev_chunk_left_over"] is None
    assert all(call["prev_chunk_left_over"] is not None for call in policy_local.calls[1:])


def test_float32_wire_dtype_is_byte_exact():
    """Round-tripping non-dyadic float32 values through the tensor codec
    must reproduce the exact bytes (no dtype casts, no re-quantization)."""
    rng = np.random.default_rng(7)
    arr = (rng.standard_normal((CHUNK_SIZE, STATE_DIM)) * 0.1).astype(np.float32)
    decoded = codec.decode_tensor(codec.encode_tensor(arr))
    assert decoded.dtype == np.float32
    assert decoded.shape == arr.shape
    assert decoded.tobytes() == arr.tobytes()
    assert torch.equal(torch.from_numpy(decoded), torch.from_numpy(arr))
