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

"""Actionable errors when a policy checkpoint's Hub repository does not exist.

Regression test for https://github.com/huggingface/lerobot/issues/4919: loading
`lerobot/fineart_vla_base` (a repo that has not been released) surfaced as a raw
HuggingFace Hub 401/404 or a confusing "config.json not found" error, with no hint that
the whole repository is missing or that a PI0.5 fallback exists.
"""

import pytest

import lerobot.configs.policies as cp
import lerobot.policies.pretrained as pp
from lerobot.utils.hub import missing_hub_repo_message


class _FakeRepoNotFound(cp.RepositoryNotFoundError):
    """RepositoryNotFoundError that can be raised without a real HTTP response object."""

    def __init__(self):
        pass


def test_missing_hub_repo_message_generic():
    message = missing_hub_repo_message("user/does-not-exist", "config.json")
    assert "user/does-not-exist" in message
    assert "does not exist" in message
    assert "config.json" in message


def test_missing_hub_repo_message_fineart_fallback():
    message = missing_hub_repo_message("lerobot/fineart_vla_base", "config.json")
    assert "lerobot/fineart_vla_base" in message
    assert "lerobot/pi05_base" in message
    assert "fineart_vla" in message


def test_config_from_pretrained_missing_repo_is_actionable(monkeypatch):
    """A missing repo at config-load time raises FileNotFoundError pointing to the fallback."""

    def fake_hf_hub_download(**kwargs):
        raise _FakeRepoNotFound()

    monkeypatch.setattr(cp, "hf_hub_download", fake_hf_hub_download)

    with pytest.raises(FileNotFoundError, match="lerobot/pi05_base"):
        cp.PreTrainedConfig.from_pretrained("lerobot/fineart_vla_base")


def test_policy_from_pretrained_missing_repo_is_actionable(monkeypatch):
    """A missing repo at weight-download time raises FileNotFoundError pointing to the fallback."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pretrained import PreTrainedPolicy

    class _StubConfig(PreTrainedConfig):
        @property
        def observation_delta_indices(self):
            return None

        @property
        def action_delta_indices(self):
            return None

        @property
        def reward_delta_indices(self):
            return None

        def validate_features(self):
            pass

        def get_optimizer_preset(self):
            return None

        def get_scheduler_preset(self):
            return None

    class _StubPolicy(PreTrainedPolicy):
        config_class = _StubConfig
        name = "stub"

        def __init__(self, config):
            super().__init__(config)
            self.config = config

        def get_optim_params(self):
            return {}

        def reset(self):
            pass

        def select_action(self, batch):
            raise NotImplementedError

        def predict_action_chunk(self, batch):
            raise NotImplementedError

        def forward(self, batch):
            raise NotImplementedError

        @classmethod
        def _load_as_safetensor(cls, model, model_file, map_location, strict):
            return model

    def fake_hf_hub_download(**kwargs):
        raise _FakeRepoNotFound()

    monkeypatch.setattr(pp, "hf_hub_download", fake_hf_hub_download)

    with pytest.raises(FileNotFoundError, match="lerobot/pi05_base"):
        _StubPolicy.from_pretrained("lerobot/fineart_vla_base", config=_StubConfig())
