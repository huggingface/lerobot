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
"""
Once you have trained a policy with our training script (lerobot/scripts/train.py), use this script to push it
to the hub.

Example:

```bash
python lerobot/scripts/push_pretrained.py \
    --pretrained_path=outputs/train/act_aloha_sim_transfer_cube_human/checkpoints/last/pretrained_model \
    --repo_id=lerobot/act_aloha_sim_transfer_cube_human
```
"""

from dataclasses import dataclass
from pathlib import Path

import draccus
from huggingface_hub import HfApi


@dataclass
class PushPreTrainedConfig:
    pretrained_path: Path
    repo_id: str
    branch: str | None = None
    private: bool = False
    exist_ok: bool = False


@draccus.wrap()
def main(cfg: PushPreTrainedConfig):
    hub_api = HfApi()
    hub_api.create_repo(
        repo_id=cfg.repo_id,
        private=cfg.private,
        repo_type="model",
        exist_ok=cfg.exist_ok,
    )
    if cfg.branch:
        hub_api.create_branch(
            repo_id=cfg.repo_id,
            branch=cfg.branch,
            repo_type="model",
            exist_ok=cfg.exist_ok,
        )

    hub_api.upload_folder(
        repo_id=cfg.repo_id,
        folder_path=cfg.pretrained_path,
        repo_type="model",
        revision=cfg.branch,
    )


if __name__ == "__main__":
    main()
