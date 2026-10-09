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

import numpy as np
import pytest

from lerobot.transport.wire.features import FeatureSpec, feature_mismatch, validate_array
from lerobot.transport.wire.protocol import (
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    correlate_reply,
    validate_reply,
)


def request():
    return Envelope(MessageType.CONTROL, "instance", "session", 4, "request")


@pytest.mark.parametrize(
    "field,value",
    [("instance_id", "other"), ("session_id", "other"), ("request_id", "other"), ("generation", 5)],
)
def test_reply_identity_is_checked_before_error_body(field, value):
    source = request()
    response = replace(source.reply(MessageType.ERROR, {"code": "invalid"}), **{field: value})
    with pytest.raises(ProtocolError) as exc:
        validate_reply(response, source, MessageType.ACK)
    assert exc.value.code is ErrorCode.STALE


def test_discovery_and_admission_allow_only_negotiated_identifiers():
    source = Envelope(MessageType.DESCRIBE, request_id="request")
    response = replace(source.reply(MessageType.DESCRIPTOR, {}), instance_id="new-instance")
    validate_reply(response, source, MessageType.DESCRIPTOR, check_instance=False)
    source = Envelope(MessageType.OPEN, "instance", request_id="request")
    response = replace(source.reply(MessageType.ACCEPTED, {}), session_id="new-session")
    validate_reply(response, source, MessageType.ACCEPTED, check_session=False)
    with pytest.raises(ProtocolError) as exc:
        correlate_reply(replace(response, generation=1), source, check_session=False)
    assert exc.value.code is ErrorCode.STALE


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"code": "unknown"},
        {"code": []},
        {"code": "execution", "message": []},
        {"code": "execution", "message": "bad", "details": []},
    ],
)
def test_malformed_error_bodies(body):
    source = request()
    with pytest.raises(ProtocolError) as exc:
        validate_reply(source.reply(MessageType.ERROR, body), source, MessageType.ACK)
    assert exc.value.code is ErrorCode.MALFORMED


def test_structured_peer_error_preserves_details():
    source = request()
    with pytest.raises(ProtocolError) as exc:
        validate_reply(
            source.error(ErrorCode.BUSY, "owned", details={"owner": "other"}), source, MessageType.ACK
        )
    assert exc.value.code is ErrorCode.BUSY
    assert str(exc.value) == "owned"
    assert exc.value.details == {"owner": "other"}


def test_unexpected_successful_reply_type():
    source = request()
    with pytest.raises(ProtocolError) as exc:
        validate_reply(source.reply(MessageType.ACTION, {}), source, MessageType.ACK)
    assert exc.value.code is ErrorCode.MALFORMED


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "other"},
        {"shape": (3,), "names": ()},
        {"dtype": "float64"},
        {"names": ("y", "x")},
        {"semantics": "position-v1"},
    ],
)
def test_feature_comparison_identifies_incompatible_fields(changes):
    expected = FeatureSpec("action", (2,), "float32", names=("x", "y"), semantics="delta-v1")
    assert feature_mismatch(expected, expected) is None
    mismatch = feature_mismatch(replace(expected, **changes), expected)
    assert mismatch is not None
    assert next(iter(changes)) in mismatch


def test_feature_comparison_checks_modality():
    expected = FeatureSpec("image", (2, 2, 3), "uint8", kind="rgb", semantics="rgb-v1")
    assert "kind" in feature_mismatch(replace(expected, kind="tensor"), expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [1, 2],
        np.zeros((2,), np.float32),
        np.zeros((3, 2), np.float64),
        np.full((3, 2), np.nan, np.float32),
    ],
)
def test_array_validation_checks_batch_dtype_and_finiteness(value):
    feature = FeatureSpec("state", (2,), "float32", semantics="state-v1")
    validate_array(np.zeros((3, 2), np.float32), feature, leading_shape=(3,))
    with pytest.raises(ValueError):
        validate_array(value, feature, leading_shape=(3,))


def test_profile_aliases_match_remote_feature_contract():
    from types import SimpleNamespace

    from lerobot.env_server.contracts import EnvDescriptor
    from lerobot.env_server.profiles import EvalProfile
    from lerobot.remote_inference.client import _compare_feature

    camera = FeatureSpec("observation.images.native", (2, 2, 3), "uint8", kind="rgb", semantics="delta-v1")
    action = FeatureSpec("action", (2,), "float32", names=("dx", "dy"), semantics="delta-v1")
    descriptor = EnvDescriptor(
        "toy",
        "robot",
        "delta-v1",
        (camera,),
        action,
        "delta",
        "command",
        20,
        ("lockstep",),
        {"reach": ["0"]},
        10,
        {},
    )
    profile = EvalProfile(
        "delta-v1",
        feature_mapping={camera.name: "observation.images.image"},
        action_name_mapping={"dx": "action_0", "dy": "action_1"},
    )
    checkpoint = SimpleNamespace(
        input_features={"observation.images.image": SimpleNamespace(shape=(3, 2, 2))},
        output_features={"action": SimpleNamespace(shape=(2,))},
        action_feature_names=["action_0", "action_1"],
    )
    profile.validate_policy(descriptor, checkpoint)
    expected_camera = replace(camera, name="observation.images.image")
    expected_action = replace(action, names=("action_0", "action_1"))
    with pytest.raises(ProtocolError) as exc:
        _compare_feature(camera, expected_camera)
    assert exc.value.code is ErrorCode.INCOMPATIBLE
    _compare_feature(replace(camera, name=profile.feature_mapping[camera.name]), expected_camera)
    _compare_feature(replace(action, names=profile.policy_action_names(descriptor)), expected_action)
    checkpoint.action_feature_names.reverse()
    with pytest.raises(ValueError, match="component order"):
        profile.validate_policy(descriptor, checkpoint)
    with pytest.raises(ProtocolError) as exc:
        _compare_feature(
            replace(expected_action, names=tuple(checkpoint.action_feature_names)), expected_action
        )
    assert exc.value.code is ErrorCode.INCOMPATIBLE


@pytest.mark.parametrize("service", ["sim", "inference"])
@pytest.mark.parametrize("failure", [None, "request_id", "generation", "malformed_error", "wrong_type"])
def test_both_clients_use_the_shared_reply_rules(service, failure):
    from types import SimpleNamespace

    from lerobot.transport.wire.codec import decode_message, encode_message

    def query(_key, payload, *_args, **_kwargs):
        source = decode_message(payload)
        response = source.reply(MessageType.ACK, {"ok": True})
        if failure == "request_id":
            response = replace(response, request_id="obsolete")
        elif failure == "generation":
            response = replace(response, generation=source.generation + 1)
        elif failure == "malformed_error":
            response = source.reply(MessageType.ERROR, {"code": []})
        elif failure == "wrong_type":
            response = source.reply(MessageType.ACTION, {})
        return [encode_message(response)]

    if service == "sim":
        from lerobot.env_server.client import EnvClient

        client = EnvClient.__new__(EnvClient)
        client.instance, client.session, client.generation, client.timeout_s = "instance", "session", 4, 1
        client.transport = SimpleNamespace(query=query)

        def invoke():
            return client._query("key", MessageType.CONTROL, {})
    else:
        from lerobot.remote_inference.client import RemoteClient

        client = RemoteClient.__new__(RemoteClient)
        client.transport = SimpleNamespace(query=query)

        def invoke():
            return client._query("key", request(), 1, MessageType.ACK).body

    if failure is None:
        assert invoke() == {"ok": True}
    else:
        with pytest.raises(ProtocolError) as exc:
            invoke()
        assert exc.value.code is (
            ErrorCode.STALE if failure in {"request_id", "generation"} else ErrorCode.MALFORMED
        )
