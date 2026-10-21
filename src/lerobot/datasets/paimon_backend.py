#!/usr/bin/env python

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

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import torch

from lerobot.utils.import_utils import _pypaimon_available, require_package

if TYPE_CHECKING:
    from .dataset_metadata import LeRobotDatasetMetadata

if TYPE_CHECKING or _pypaimon_available:
    import pypaimon.multimodal as pm

from .dataset_reader import BaseDatasetReader
from .io_utils import write_episodes, write_stats, write_tasks
from .video_utils import apply_rgb_transforms


def prepare_paimon_dataset(
    root: str | Path,
    *,
    table: str,
    catalog_options: dict[str, str],
    database: str = "default",
    tag_name: str,
) -> Path:
    """Create a local LeRobot metadata directory pointing to a tagged Paimon table group.

    Only metadata is exported; frame rows and media remain in Paimon. Credentials
    should come from the runtime environment, not ``catalog_options`` (saved to disk).
    """
    require_package("pypaimon", extra="paimon")
    if not tag_name:
        raise ValueError("A LeRobot table-group tag is required.")
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    config = {"table": table, "catalog_options": catalog_options, "database": database, "tag_name": tag_name}
    conn = pm.connect(database=database, options=catalog_options)
    dataset = pm.PaimonLeRobotDataset(conn.get_table(table), tag_name=tag_name)
    try:
        if dataset.meta.subtasks is not None:
            raise NotImplementedError("Paimon subtask metadata is not supported by this adapter yet.")
        info = dict(dataset.meta.info, storage_format="paimon")
        (root / "meta").mkdir(parents=True)
        (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        write_tasks(dataset.meta.tasks, root)
        write_episodes(dataset.meta.episodes, root)
        if dataset.meta.stats is not None:
            write_stats(dataset.meta.stats, root)
        (root / "paimon.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    finally:
        dataset.close()
    return root


def localize_root(
    repo_id: str | None,
    root: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> NoReturn:
    raise FileNotFoundError("Paimon uses a local metadata directory; its catalog may be remote.")


class PaimonDatasetReader(BaseDatasetReader):
    """Delegate map-style reads to PyPaimon while retaining LeRobot's public dataset API."""

    def __init__(
        self,
        *,
        meta: LeRobotDatasetMetadata,
        root: str | Path | None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        return_uint8: bool = False,
        depth_output_unit: str = "mm",
        token: str | bool | None = None,
    ) -> None:
        require_package("pypaimon", extra="paimon")
        if meta.depth_keys or meta.has_language_columns:
            raise NotImplementedError("Paimon currently supports RGB, state/action and task text datasets.")
        if root is None:
            raise ValueError("Paimon requires root pointing to a prepared local metadata directory.")
        config = json.loads((Path(root) / "paimon.json").read_text(encoding="utf-8"))
        if not config.get("tag_name"):
            raise ValueError("A LeRobot table-group tag is required.")
        conn = pm.connect(database=config["database"], options=config["catalog_options"])
        self._dataset = pm.PaimonLeRobotDataset(
            conn.get_table(config["table"]),
            tag_name=config["tag_name"],
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            return_uint8=return_uint8,
        )
        # Reject stale metadata rather than train with incorrect normalization or boundaries.
        info = dict(self._dataset.meta.info, storage_format="paimon")
        stored_info = json.loads((Path(root) / "meta/info.json").read_text(encoding="utf-8"))
        if json.loads(json.dumps(info)) != stored_info:
            self._dataset.close()
            raise ValueError("Paimon metadata changed; prepare a new directory for the selected tag.")
        self.episodes = self._dataset.episodes
        self._image_keys = meta.image_keys
        self._camera_keys = meta.camera_keys
        self.set_image_transforms(image_transforms)

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._dataset.absolute_to_relative_idx

    def get_item(self, idx: int) -> dict:
        return self.get_items([idx])[0]

    def get_items(self, indices: list[int]) -> list[dict]:
        items = self._dataset.__getitems__(indices)
        for item in items:
            # LeRobot keeps embedded images normalized even with return_uint8=True.
            for key in self._image_keys:
                if item[key].dtype == torch.uint8:
                    item[key] = item[key].float().div_(255)
            apply_rgb_transforms(item, self._image_transforms, self._camera_keys, [])
        return items

    def close(self) -> None:
        self._dataset.close()


DATASET_READER = PaimonDatasetReader
