"""SARM training launcher that skips video decode in __getitem__.

When the SARM CLIP cache is enabled, the LeRobotDataset still decodes H.264
video frames per sample then those tensors get discarded by
SARMEncodingProcessorStep. At 224x224 this dominates step time. This wrapper
monkey-patches LeRobotDataset.__getitem__ to short-circuit the video query —
the SARM encoder uses cached features keyed by global frame index instead.

Usage (drop-in replacement for `lerobot-train` for SARM cfgs only):

    uv run python scripts_local/sarm_train_no_video.py \
        --config_path=src/lerobot/rl/sim_3stage_sarm_v3_iter1_train.json
"""
from __future__ import annotations
import sys
import logging

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _patched_getitem(self, idx):
    """Drop-in replacement that skips video frame decoding.

    Returns the same dict structure as the original __getitem__ but with
    image keys filled with a tiny zero placeholder. SARMEncodingProcessorStep
    overwrites these via clip_cache lookup keyed by global frame index, so
    the actual pixel values are never used.
    """
    self._ensure_hf_dataset_loaded()
    item = self.hf_dataset[idx]
    ep_idx = item["episode_index"].item()
    abs_idx = item["index"].item()

    query_indices = None
    if self.delta_indices is not None:
        query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
        query_result = self._query_hf_dataset(query_indices)
        item = {**item, **padding}
        for key, val in query_result.items():
            item[key] = val

    # Skip video decode entirely — fill with placeholder zeros.
    if len(self.meta.video_keys) > 0:
        import torch
        for vid_key in self.meta.video_keys:
            shape = self.meta.info["features"][vid_key]["shape"]
            if query_indices is not None and vid_key in query_indices:
                n_frames = len(query_indices[vid_key])
                item[vid_key] = torch.zeros((n_frames, *shape), dtype=torch.uint8)
            else:
                item[vid_key] = torch.zeros(shape, dtype=torch.uint8)

    if self.image_transforms is not None:
        image_keys = self.meta.camera_keys
        for cam in image_keys:
            item[cam] = self.image_transforms(item[cam])

    task_idx = item["task_index"].item()
    item["task"] = self.meta.tasks.iloc[task_idx].name

    if "subtask_index" in self.features and self.meta.subtasks is not None:
        subtask_idx = item["subtask_index"].item()
        item["subtask"] = self.meta.subtasks.iloc[subtask_idx].name

    return item


def _patch():
    LeRobotDataset.__getitem__ = _patched_getitem
    logging.getLogger(__name__).warning(
        "[SARM] LeRobotDataset.__getitem__ patched to skip video decode. "
        "Cached CLIP features must be loaded by SARMEncodingProcessorStep."
    )


if __name__ == "__main__":
    _patch()
    from lerobot.scripts.lerobot_train import main
    main()
