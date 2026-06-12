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

"""Unit tests for the wire schema: packed header and key-expression layout."""

import pytest

from lerobot.policy_server.schema import (
    HEADER_SIZE,
    MSG_TYPE_CHUNK,
    MSG_TYPE_EVENT,
    MSG_TYPE_OBS,
    RESERVED_SEGMENTS,
    SCHEMA_VERSION,
    MsgHeader,
    action_key,
    client_alive_key,
    client_alive_wildcard,
    client_uuid_from_key,
    obs_key,
    obs_wildcard,
    reset_key,
    reset_wildcard,
    sanitize_key_segment,
    server_alive_key,
    service_prefix,
    session_key,
    status_key,
)

# ---------------------------------------------------------------------------
# MsgHeader
# ---------------------------------------------------------------------------


def test_header_roundtrip_all_fields():
    hdr = MsgHeader(
        schema_version=3,
        msg_type=MSG_TYPE_CHUNK,
        seq_id=123456789,
        episode_id=42,
        client_mono_ns=987654321012345,
        session_epoch=7,
    )
    out = MsgHeader.unpack(hdr.pack())
    assert out == hdr


def test_header_defaults_roundtrip():
    out = MsgHeader.unpack(MsgHeader().pack())
    assert out.schema_version == SCHEMA_VERSION
    assert out.msg_type == MSG_TYPE_OBS
    assert out.seq_id == 0
    assert out.episode_id == 0
    assert out.client_mono_ns == 0
    assert out.session_epoch == 0


def test_header_negative_client_mono_ns():
    hdr = MsgHeader(msg_type=MSG_TYPE_EVENT, client_mono_ns=-123456789)
    out = MsgHeader.unpack(hdr.pack())
    assert out.client_mono_ns == -123456789


def test_header_max_u64_seq_id():
    max_u64 = 2**64 - 1
    hdr = MsgHeader(seq_id=max_u64)
    out = MsgHeader.unpack(hdr.pack())
    assert out.seq_id == max_u64


def test_header_size_constant_matches_pack():
    assert len(MsgHeader().pack()) == HEADER_SIZE


def test_header_unpack_rejects_wrong_length():
    packed = MsgHeader().pack()
    with pytest.raises(ValueError, match="Bad header length"):
        MsgHeader.unpack(packed[:-1])
    with pytest.raises(ValueError, match="Bad header length"):
        MsgHeader.unpack(packed + b"\x00")
    with pytest.raises(ValueError, match="Bad header length"):
        MsgHeader.unpack(b"")


# ---------------------------------------------------------------------------
# sanitize_key_segment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_char", ["/", "*", "$", "?", "#", " "])
def test_sanitize_folds_unsafe_chars_to_dash(bad_char):
    assert sanitize_key_segment(f"a{bad_char}b") == "a-b"


def test_sanitize_folds_runs_to_single_dash():
    assert sanitize_key_segment("a/*$?# b") == "a-b"


def test_sanitize_strips_whitespace_and_edge_dashes():
    assert sanitize_key_segment("  hello  ") == "hello"
    assert sanitize_key_segment("/leading/trailing/") == "leading-trailing"


def test_sanitize_preserves_allowed_chars():
    assert sanitize_key_segment("Az09_.-x") == "Az09_.-x"


@pytest.mark.parametrize("empty_input", ["", "   ", "***", "/?#$*"])
def test_sanitize_raises_on_empty_after_sanitize(empty_input):
    with pytest.raises(ValueError, match="empty"):
        sanitize_key_segment(empty_input)


@pytest.mark.parametrize(
    "reserved", sorted(["status", "session", "server", "obs", "action", "reset", "alive"])
)
def test_sanitize_raises_on_reserved_segments(reserved):
    assert reserved in RESERVED_SEGMENTS
    with pytest.raises(ValueError, match="reserved"):
        sanitize_key_segment(reserved)


# ---------------------------------------------------------------------------
# service_prefix
# ---------------------------------------------------------------------------


def test_service_prefix_example():
    prefix = service_prefix("lerobot/pi0_towels", "main", "fold the towel")
    assert prefix == "@lerobot/lerobot-pi0_towels/main/fold-the-towel"


def test_service_prefix_defaults_for_empty_revision_and_task():
    prefix = service_prefix("org/model", "", "")
    assert prefix == "@lerobot/org-model/main/default"


# ---------------------------------------------------------------------------
# Key builders and wildcards
# ---------------------------------------------------------------------------

PREFIX = "@lerobot/org-model/main/default"
UUID = "client-uuid-1234"


def test_per_client_keys():
    assert obs_key(PREFIX, UUID) == f"{PREFIX}/{UUID}/obs"
    assert action_key(PREFIX, UUID) == f"{PREFIX}/{UUID}/action"
    assert reset_key(PREFIX, UUID) == f"{PREFIX}/{UUID}/reset"
    assert client_alive_key(PREFIX, UUID) == f"{PREFIX}/{UUID}/alive"


def test_service_level_keys():
    assert status_key(PREFIX) == f"{PREFIX}/status"
    assert session_key(PREFIX) == f"{PREFIX}/session"
    assert server_alive_key(PREFIX) == f"{PREFIX}/server/alive"


def test_wildcards_are_single_depth():
    assert obs_wildcard(PREFIX) == f"{PREFIX}/*/obs"
    assert reset_wildcard(PREFIX) == f"{PREFIX}/*/reset"
    assert client_alive_wildcard(PREFIX) == f"{PREFIX}/*/alive"
    assert "**" not in obs_wildcard(PREFIX)


def test_key_builders_sanitize_client_uuid():
    assert obs_key(PREFIX, "bad uuid") == f"{PREFIX}/bad-uuid/obs"
    with pytest.raises(ValueError):
        obs_key(PREFIX, "status")


# ---------------------------------------------------------------------------
# client_uuid_from_key
# ---------------------------------------------------------------------------


def test_client_uuid_from_obs_reset_alive_keys():
    assert client_uuid_from_key(obs_key(PREFIX, UUID)) == UUID
    assert client_uuid_from_key(reset_key(PREFIX, UUID)) == UUID
    assert client_uuid_from_key(client_alive_key(PREFIX, UUID)) == UUID


def test_client_uuid_from_key_rejects_keys_without_client_chunk():
    with pytest.raises(ValueError, match="no client chunk"):
        client_uuid_from_key("obs")
