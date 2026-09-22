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

import json
import sys
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies import factory  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.processor import (  # noqa: E402
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.scripts import lerobot_eval  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE  # noqa: E402


def test_eval_without_cli_rename_map_uses_saved_camera_names(tmp_path, monkeypatch):
    camera = f"{OBS_IMAGES}.camera1"
    config = ACTConfig(
        device="cpu",
        input_features={
            camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 384, 384)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    config.save_pretrained(tmp_path)
    PolicyProcessorPipeline(
        [RenameObservationsProcessorStep(rename_map={OBS_IMAGE: camera}), DeviceProcessorStep(device="cpu")]
    ).save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    PolicyProcessorPipeline([DeviceProcessorStep(device="cpu")]).save_pretrained(
        tmp_path, config_filename="policy_postprocessor.json"
    )

    policy = torch.nn.Linear(2, 2)
    policy.config = config
    policy_class = MagicMock()
    policy_class.from_pretrained.return_value = policy
    monkeypatch.setattr(factory, "get_policy_class", lambda _: policy_class)
    monkeypatch.setattr(lerobot_eval, "make_env", lambda *args, **kwargs: {})

    def eval_policy_all(*, preprocessor, **kwargs):
        image = torch.ones(1, 3, 384, 384)
        observation = preprocessor({OBS_IMAGE: image})
        assert OBS_IMAGE not in observation
        torch.testing.assert_close(observation[camera], image)
        return {"overall": {"pc_success": 100.0}}

    monkeypatch.setattr(lerobot_eval, "eval_policy_all", eval_policy_all)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lerobot-eval", f"--policy.path={tmp_path}", "--env.type=pusht", f"--output_dir={tmp_path}"],
    )

    lerobot_eval.eval_main()

    assert json.loads((tmp_path / "eval_info.json").read_text()) == {"overall": {"pc_success": 100.0}}
