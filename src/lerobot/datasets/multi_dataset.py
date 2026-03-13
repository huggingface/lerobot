"""MultiLeRobotDataset: joint training over heterogeneous LeRobot datasets.

Supports:
- Per-dataset feature mapping (rename keys to a unified namespace)
- Automatic zero-padding for features missing in some datasets
- Per-dataset transform pipelines
- Weighted sampling via dataset weights
- Aggregated stats across all sub-datasets
- A ``meta`` shim compatible with EpisodeAwareSampler and make_policy
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch
import torch.utils.data

from lerobot.configs.default import SubDatasetConfig
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import DatasetTransformPipeline


class MultiDatasetMeta:
    """Lightweight metadata shim that exposes the same interface as ``LeRobotDatasetMetadata``.

    Built by aggregating the metadata of multiple sub-datasets after their
    feature keys have been mapped to a unified namespace.
    """

    def __init__(
        self,
        datasets: list[LeRobotDataset],
        feature_maps: list[dict[str, str]],
    ):
        self._datasets = datasets
        self._feature_maps = feature_maps

        self._unified_features = self._build_unified_features()
        self._episodes = self._build_episodes()
        self._stats = self._build_stats()

    # ------------------------------------------------------------------
    # Feature union
    # ------------------------------------------------------------------

    def _build_unified_features(self) -> dict[str, dict]:
        """Build feature dict as the *union* of all mapped feature keys."""
        unified: dict[str, dict] = {}
        for ds, fmap in zip(self._datasets, self._feature_maps):
            for original_key, feat_info in ds.meta.features.items():
                mapped_key = fmap.get(original_key, original_key)
                if mapped_key not in unified:
                    unified[mapped_key] = dict(feat_info)
                else:
                    existing_shape = tuple(unified[mapped_key]["shape"])
                    new_shape = tuple(feat_info["shape"])
                    if existing_shape != new_shape and unified[mapped_key]["dtype"] == feat_info["dtype"]:
                        logging.warning(
                            "Feature '%s' has shape %s in one dataset but %s in another. "
                            "The larger shape will be used (padding applied automatically).",
                            mapped_key,
                            existing_shape,
                            new_shape,
                        )
                        if np.prod(new_shape) > np.prod(existing_shape):
                            unified[mapped_key] = dict(feat_info)
        return unified

    # ------------------------------------------------------------------
    # Episode metadata (global flat indexing)
    # ------------------------------------------------------------------

    def _build_episodes(self) -> dict[str, list]:
        """Concatenate episode boundaries across sub-datasets with frame offsets.

        Produces the same column structure as ``load_episodes()`` so that
        ``EpisodeAwareSampler`` and ``WeightedEpisodeAwareSampler`` can consume it.
        """
        from_indices: list[int] = []
        to_indices: list[int] = []
        dataset_source: list[int] = []

        frame_offset = 0
        for ds_idx, ds in enumerate(self._datasets):
            eps = ds.meta.episodes
            for ep in eps:
                from_indices.append(ep["dataset_from_index"] + frame_offset)
                to_indices.append(ep["dataset_to_index"] + frame_offset)
                dataset_source.append(ds_idx)
            frame_offset += ds.num_frames

        return {
            "dataset_from_index": from_indices,
            "dataset_to_index": to_indices,
            "dataset_source": dataset_source,
        }

    # ------------------------------------------------------------------
    # Stats aggregation
    # ------------------------------------------------------------------

    def _build_stats(self) -> dict[str, dict[str, np.ndarray]]:
        """Aggregate stats across sub-datasets using mapped feature keys."""
        mapped_stats_list: list[dict[str, dict]] = []
        for ds, fmap in zip(self._datasets, self._feature_maps):
            reverse_map = {v: k for k, v in fmap.items()}
            mapped: dict[str, dict] = {}
            for unified_key in self._unified_features:
                original_key = reverse_map.get(unified_key, unified_key)
                if original_key in ds.meta.stats:
                    mapped[unified_key] = ds.meta.stats[original_key]
            mapped_stats_list.append(mapped)

        return aggregate_stats(mapped_stats_list)

    # ------------------------------------------------------------------
    # Properties matching LeRobotDatasetMetadata API
    # ------------------------------------------------------------------

    @property
    def features(self) -> dict[str, dict]:
        return self._unified_features

    @property
    def image_keys(self) -> list[str]:
        return [k for k, f in self._unified_features.items() if f["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [k for k, f in self._unified_features.items() if f["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [k for k, f in self._unified_features.items() if f["dtype"] in ("video", "image")]

    @property
    def names(self) -> dict[str, list | dict]:
        return {k: f["names"] for k, f in self._unified_features.items()}

    @property
    def shapes(self) -> dict[str, tuple]:
        return {k: tuple(f["shape"]) for k, f in self._unified_features.items()}

    @property
    def fps(self) -> int:
        fps_values = {ds.meta.fps for ds in self._datasets}
        if len(fps_values) > 1:
            logging.warning("Sub-datasets have different FPS values: %s. Using the first.", fps_values)
        return self._datasets[0].meta.fps

    @property
    def stats(self) -> dict[str, dict[str, np.ndarray]]:
        return self._stats

    @stats.setter
    def stats(self, value: dict):
        self._stats = value

    @property
    def episodes(self) -> dict[str, list]:
        return self._episodes

    @property
    def total_episodes(self) -> int:
        return sum(ds.meta.total_episodes for ds in self._datasets)

    @property
    def total_frames(self) -> int:
        return sum(ds.meta.total_frames for ds in self._datasets)

    @property
    def total_tasks(self) -> int:
        return sum(ds.meta.total_tasks for ds in self._datasets)

    @property
    def info(self) -> dict:
        return {
            "fps": self.fps,
            "features": self._unified_features,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "total_tasks": self.total_tasks,
            "codebase_version": "v3.0",
        }


class NewMultiLeRobotDataset(torch.utils.data.Dataset):
    """Dataset that wraps multiple ``LeRobotDataset`` instances with feature mapping and padding.

    Each sub-dataset can have different feature names and shapes.  A per-dataset
    ``feature_map`` renames keys into a shared namespace.  Features that a given
    sub-dataset does not provide are zero-padded so every ``__getitem__`` returns
    the full unified feature set.
    """

    def __init__(
        self,
        configs: list[SubDatasetConfig],
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
    ):
        super().__init__()
        self._configs = configs
        self.image_transforms = image_transforms

        self._datasets: list[LeRobotDataset] = []
        self._feature_maps: list[dict[str, str]] = []
        self._transform_pipelines: list[DatasetTransformPipeline | None] = []
        self._weights: list[float] = []

        for cfg in configs:
            ds = LeRobotDataset(
                repo_id=cfg.repo_id,
                root=cfg.root,
                episodes=cfg.episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                revision=cfg.revision,
                video_backend=cfg.video_backend,
            )
            self._datasets.append(ds)
            self._feature_maps.append(cfg.feature_map or {})
            self._transform_pipelines.append(
                DatasetTransformPipeline(cfg.transforms) if cfg.transforms else None
            )
            self._weights.append(cfg.weight)

        self._meta = MultiDatasetMeta(self._datasets, self._feature_maps)

        # Pre-compute cumulative frame counts for fast index mapping.
        self._cumulative_frames: list[int] = []
        total = 0
        for ds in self._datasets:
            total += ds.num_frames
            self._cumulative_frames.append(total)

        # Build reverse maps (unified_key -> original_key) per dataset for padding.
        self._reverse_maps: list[dict[str, str]] = []
        for fmap in self._feature_maps:
            self._reverse_maps.append({v: k for k, v in fmap.items()})

        logging.info(
            "MultiLeRobotDataset: %d sub-datasets, %d total frames, %d total episodes, "
            "%d unified features",
            len(self._datasets),
            self.num_frames,
            self.num_episodes,
            len(self._meta.features),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def meta(self) -> MultiDatasetMeta:
        return self._meta

    @property
    def dataset_weights(self) -> list[float]:
        return self._weights

    @property
    def num_frames(self) -> int:
        return self._cumulative_frames[-1] if self._cumulative_frames else 0

    @property
    def num_episodes(self) -> int:
        return sum(ds.num_episodes for ds in self._datasets)

    @property
    def episodes(self) -> list[int] | None:
        return None

    @property
    def fps(self) -> int:
        return self._meta.fps

    @property
    def features(self) -> dict[str, dict]:
        return self._meta.features

    @property
    def camera_keys(self) -> list[str]:
        return self._meta.camera_keys

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map a global frame index to (dataset_index, local_index)."""
        for ds_idx, cum in enumerate(self._cumulative_frames):
            if idx < cum:
                local = idx - (self._cumulative_frames[ds_idx - 1] if ds_idx > 0 else 0)
                return ds_idx, local
        raise IndexError(f"Index {idx} out of range (total {self.num_frames})")

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ds_idx, local_idx = self._locate(idx)
        item = self._datasets[ds_idx][local_idx]

        # 1. Rename keys according to feature_map.
        fmap = self._feature_maps[ds_idx]
        if fmap:
            renamed: dict[str, torch.Tensor] = {}
            for key, value in item.items():
                renamed[fmap.get(key, key)] = value
            item = renamed

        # 2. Apply per-dataset transform pipeline.
        pipeline = self._transform_pipelines[ds_idx]
        if pipeline is not None:
            item = pipeline(item)

        # 3. Pad missing features with zeros.
        reverse_map = self._reverse_maps[ds_idx]
        ds_features = self._datasets[ds_idx].meta.features
        for unified_key, feat_info in self._meta.features.items():
            if unified_key in item:
                continue
            original_key = reverse_map.get(unified_key, unified_key)
            if original_key in ds_features:
                continue
            shape = tuple(feat_info["shape"])
            dtype = feat_info["dtype"]
            if dtype in ("video", "image"):
                # Camera tensors are (C, H, W) after transforms.
                c, h, w = (shape[2], shape[0], shape[1]) if len(shape) == 3 else (3, shape[0], shape[1])
                item[unified_key] = torch.zeros(c, h, w, dtype=torch.float32)
            elif dtype in ("float32", "float64"):
                item[unified_key] = torch.zeros(shape, dtype=torch.float32)
            elif dtype in ("int32", "int64"):
                item[unified_key] = torch.zeros(shape, dtype=torch.int64)
            elif dtype == "bool":
                item[unified_key] = torch.zeros(shape, dtype=torch.bool)
            else:
                item[unified_key] = torch.zeros(shape, dtype=torch.float32)
            item[f"{unified_key}_is_pad"] = torch.tensor(True)

        # 4. Tag which dataset this sample came from.
        item["dataset_index"] = torch.tensor(ds_idx)
        return item

    def __repr__(self) -> str:
        repo_ids = [c.repo_id for c in self._configs]
        return (
            f"NewMultiLeRobotDataset(\n"
            f"  repo_ids={repo_ids},\n"
            f"  num_frames={self.num_frames},\n"
            f"  num_episodes={self.num_episodes},\n"
            f"  unified_features={list(self._meta.features.keys())},\n"
            f"  weights={self._weights},\n"
            f")"
        )
