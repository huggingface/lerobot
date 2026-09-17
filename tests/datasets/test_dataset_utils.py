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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from packaging.version import Version

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import httpx
from datasets import Dataset  # noqa: E402
from huggingface_hub import DatasetCard
from huggingface_hub.errors import (
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)

import lerobot.datasets.utils as dataset_utils
from lerobot.datasets.io_utils import hf_transform_to_torch
from lerobot.datasets.utils import (
    create_lerobot_dataset_card,
    get_repo_versions,
    get_safe_version,
    resolve_episode_indices,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.feature_utils import combine_feature_dicts


def calculate_episode_data_index(hf_dataset: Dataset) -> dict[str, torch.Tensor]:
    """Calculate episode data index for testing. Returns {"from": Tensor, "to": Tensor}."""
    episode_data_index: dict[str, list[int]] = {"from": [], "to": []}
    current_episode = None
    if len(hf_dataset) == 0:
        return {"from": torch.tensor([]), "to": torch.tensor([])}
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            episode_data_index["from"].append(idx)
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            current_episode = episode_idx
    episode_data_index["to"].append(idx + 1)
    return {k: torch.tensor(v) for k, v in episode_data_index.items()}


def test_default_parameters():
    card = create_lerobot_dataset_card()
    assert isinstance(card, DatasetCard)
    assert card.data.tags == ["LeRobot"]
    assert card.data.task_categories == ["robotics"]
    assert card.data.configs == [
        {
            "config_name": "default",
            "data_files": "data/*/*.parquet",
        }
    ]


def test_resolve_episode_indices_applies_allowlist_and_exclusions():
    assert resolve_episode_indices([4, 1, 3, 0], 5, [1, 4]) == [3, 0]


def test_resolve_episode_indices_preserves_none_without_filtering():
    assert resolve_episode_indices(None, 5) is None


def test_resolve_episode_indices_ignores_out_of_range_values(caplog):
    assert resolve_episode_indices([-1, 0, 3, 5], 4, [-2, 3, 8]) == [0]
    assert "Ignoring episode indices outside the dataset range [0, 4): [-1, 5]" in caplog.text
    assert "Ignoring excluded episode indices outside the dataset range [0, 4): [-2, 8]" in caplog.text


@pytest.mark.parametrize("token", ["hf_test_token", True, False])
def test_get_repo_versions_forwards_token(monkeypatch, token):
    api = Mock()
    api.list_repo_refs.return_value = SimpleNamespace(
        branches=[SimpleNamespace(name="v3.0")],
        tags=[],
    )
    hf_api = Mock(return_value=api)
    monkeypatch.setattr(dataset_utils, "HfApi", hf_api)

    assert get_repo_versions("private/repo", token=token) == [Version("3.0")]
    hf_api.assert_called_once_with(token=token)
    api.list_repo_refs.assert_called_once_with("private/repo", repo_type="dataset")


@pytest.mark.parametrize("token", ["hf_test_token", True, False])
def test_get_safe_version_forwards_token(monkeypatch, token):
    get_versions = Mock(return_value=[Version("3.0")])
    monkeypatch.setattr(dataset_utils, "get_repo_versions", get_versions)

    assert get_safe_version("private/repo", "v3.0", token=token) == "v3.0"
    get_versions.assert_called_once_with("private/repo", token=token)


def test_with_tags():
    tags = ["tag1", "tag2"]
    card = create_lerobot_dataset_card(tags=tags)
    assert card.data.tags == ["LeRobot", "tag1", "tag2"]


def test_calculate_episode_data_index():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    assert torch.equal(episode_data_index["from"], torch.tensor([0, 2, 3]))
    assert torch.equal(episode_data_index["to"], torch.tensor([2, 3, 6]))


def test_merge_simple_vectors():
    g1 = {
        ACTION: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["ee.x", "ee.y"],
        }
    }
    g2 = {
        ACTION: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["ee.y", "ee.z"],
        }
    }

    out = combine_feature_dicts(g1, g2)

    assert ACTION in out
    assert out[ACTION]["dtype"] == "float32"
    # Names merged with preserved order and de-dupuplication
    assert out[ACTION]["names"] == ["ee.x", "ee.y", "ee.z"]
    # Shape correctly recomputed from names length
    assert out[ACTION]["shape"] == (3,)


def test_merge_multiple_groups_order_and_dedup():
    g1 = {ACTION: {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}}
    g2 = {ACTION: {"dtype": "float32", "shape": (2,), "names": ["b", "c"]}}
    g3 = {ACTION: {"dtype": "float32", "shape": (3,), "names": ["a", "c", "d"]}}

    out = combine_feature_dicts(g1, g2, g3)

    assert out[ACTION]["names"] == ["a", "b", "c", "d"]
    assert out[ACTION]["shape"] == (4,)


def test_non_vector_last_wins_for_images():
    # Non-vector (images) with same name should be overwritten by the last image specified
    g1 = {
        f"{OBS_IMAGES}.front": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    }
    g2 = {
        f"{OBS_IMAGES}.front": {
            "dtype": "image",
            "shape": (3, 720, 1280),
            "names": ["channels", "height", "width"],
        }
    }

    out = combine_feature_dicts(g1, g2)
    assert out[f"{OBS_IMAGES}.front"]["shape"] == (3, 720, 1280)
    assert out[f"{OBS_IMAGES}.front"]["dtype"] == "image"


def test_dtype_mismatch_raises():
    g1 = {ACTION: {"dtype": "float32", "shape": (1,), "names": ["a"]}}
    g2 = {ACTION: {"dtype": "float64", "shape": (1,), "names": ["b"]}}

    with pytest.raises(ValueError, match="dtype mismatch for 'action'"):
        _ = combine_feature_dicts(g1, g2)


def test_non_dict_passthrough_last_wins():
    g1 = {"misc": 123}
    g2 = {"misc": 456}

    out = combine_feature_dicts(g1, g2)
    # For non-dict entries the last one wins
    assert out["misc"] == 456


def test_get_safe_version_raises_on_repo_without_version_tags(monkeypatch):
    monkeypatch.setattr(dataset_utils, "get_repo_versions", Mock(return_value=[]))

    with pytest.raises(RuntimeError, match="must be tagged with a codebase version"):
        get_safe_version("private/repo", "v3.0")


def test_get_safe_version_error_reports_repo_id(monkeypatch):
    repo_id = "private/repo"
    monkeypatch.setattr(dataset_utils, "get_repo_versions", Mock(return_value=[]))

    with pytest.raises(RuntimeError) as exc_info:
        get_safe_version(repo_id, "v3.0")

    assert repo_id in str(exc_info.value)


def _hub_http_error(status: int, reason: str) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co")
    return HfHubHTTPError(f"{status} Error: {reason}", response=httpx.Response(status, request=request))


# Factories, not instances: a raised exception keeps its ``__traceback__``, so sharing one
# across parametrized runs would accumulate state between tests.
HUB_UNUSABLE_ERRORS = {
    "offline": lambda: OfflineModeIsEnabled(
        "Cannot reach https://huggingface.co/api/datasets/private/repo/refs: offline mode is enabled."
    ),
    "connect": lambda: httpx.ConnectError("[Errno 61] Connection refused"),
    "timeout": lambda: httpx.ReadTimeout("timed out"),
    "hub-5xx": lambda: _hub_http_error(500, "Internal Server Error"),
    "hub-rate-limited": lambda: _hub_http_error(429, "Too Many Requests"),
    "repo-not-found": lambda: RepositoryNotFoundError(
        "404 Client Error. Repository Not Found",
        response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")),
    ),
}


@pytest.mark.parametrize("make_error", HUB_UNUSABLE_ERRORS.values(), ids=HUB_UNUSABLE_ERRORS)
def test_get_safe_version_keeps_requested_version_when_hub_unusable(monkeypatch, make_error):
    """The Hub cannot say which versions exist, so the requested one is passed through.

    Measured against a warm cache: these are exactly the failures `snapshot_download` absorbs to
    resolve the revision from `refs/` instead. Aborting here would fail a load the cache serves.
    """
    api = Mock()
    api.list_repo_refs.side_effect = make_error()
    monkeypatch.setattr(dataset_utils, "HfApi", Mock(return_value=api))

    # Deliberately not CODEBASE_VERSION: what comes back must be what was asked for, not the
    # version lerobot happens to default to. With "v3.0" the two are indistinguishable today.
    assert get_safe_version("private/repo", "v2.1") == "v2.1"
    api.list_repo_refs.assert_called_once_with("private/repo", repo_type="dataset")


@pytest.mark.parametrize(
    "error",
    [
        httpx.ProxyError("failed to connect to proxy"),
        httpx.RemoteProtocolError("server disconnected without sending a response"),
        httpx.UnsupportedProtocol("Request URL is missing an 'http://' protocol."),
    ],
    ids=["proxy-error", "mid-stream-drop", "no-scheme-endpoint"],
)
def test_get_safe_version_propagates_errors_that_are_not_a_hub_being_unusable(monkeypatch, error):
    """Transport failures that `snapshot_download` re-raises too, measured against a warm cache.

    Swallowing them here would not save the load: the very next call fails the same way, one frame
    further from its cause. All three are `httpx.TransportError` subclasses outside the caught
    tuple, which is why that tuple names `ConnectError` and `TimeoutException` rather than their
    shared base.
    """
    api = Mock()
    api.list_repo_refs.side_effect = error
    monkeypatch.setattr(dataset_utils, "HfApi", Mock(return_value=api))

    with pytest.raises(type(error)):
        get_safe_version("private/repo", "v2.1")


@pytest.mark.parametrize("token", [None, "hf_test_token"])
def test_get_safe_version_unusable_hub_does_not_raise_missing_version_tag(monkeypatch, token):
    """A Hub that cannot answer must not be reported as "this dataset has no version tags".

    Checked on both token paths, since each takes a different `get_repo_versions` call.
    """
    get_versions = Mock(side_effect=OfflineModeIsEnabled("offline mode is enabled"))
    monkeypatch.setattr(dataset_utils, "get_repo_versions", get_versions)

    kwargs = {} if token is None else {"token": token}
    assert get_safe_version("private/repo", Version("2.1"), **kwargs) == "v2.1"
    get_versions.assert_called_once_with("private/repo", **kwargs)
