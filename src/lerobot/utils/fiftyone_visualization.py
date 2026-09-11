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

"""FiftyOne visualization backend.

Post-hoc episode playback ([`~lerobot.utils.fiftyone_visualization.serve_fiftyone_dataset_playback`])
in the FiftyOne App. FiftyOne imports a LeRobot v3 dataset straight from its on-disk root (`meta/`,
`data/`, `videos/`) and renders each episode with per-camera video tiles and a synchronized State &
Action tile, so there is no per-frame streaming API: this backend is selectable from
`lerobot-dataset-viz` only and is not part of the live control-loop dispatch in
[`~lerobot.utils.visualization_utils`]. Requires the `fiftyone` extra (`pip install 'lerobot[fiftyone]'`).
"""

import contextlib
import logging
import re
import shlex
from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.utils import get_safe_version, is_valid_version

from .constants import HF_LEROBOT_HUB_CACHE
from .import_utils import require_package

# Video codecs the FiftyOne App decodes natively. LeRobot's default encoder (``libsvtav1``) produces
# ``av1``, so freshly recorded datasets are fine. Other codecs are handed to the browser's WebCodecs
# and may or may not play, so they are warned about up front rather than discovered as blank tiles.
_APP_NATIVE_CODECS = frozenset({"av1", "h264"})


def _fiftyone_dataset_name(repo_id: str, episode_index: int | None) -> str:
    """Default FiftyOne dataset name for a LeRobot ``repo_id`` (and optional single episode).

    FiftyOne dataset names show up in the App's dataset selector and as MongoDB collection names, so
    anything that isn't alphanumeric, ``-``, ``_`` or ``.`` is collapsed to ``-``.
    """

    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", repo_id).strip("-")
    return base if episode_index is None else f"{base}-episode-{episode_index}"


def _episode_files(meta: LeRobotDatasetMetadata, episodes: list[int] | None) -> list[Path]:
    """Relative ``data/`` and ``videos/`` paths the FiftyOne importer reads for ``episodes`` (all if ``None``)."""

    indices = range(meta.total_episodes) if episodes is None else episodes
    files: set[Path] = set()
    for ep in indices:
        files.add(meta.get_data_file_path(ep))
        for key in meta.video_keys:
            files.add(meta.get_video_file_path(ep, key))
    return sorted(files)


def _ensure_files_on_disk(
    meta: LeRobotDatasetMetadata, episodes: list[int] | None, *, root_requested: bool
) -> None:
    """Download whatever the importer needs that isn't already under ``meta.root``.

    Only the missing files are fetched, and nothing is read into memory: unlike `LeRobotDataset`, which
    loads every frame record of the selected episodes into a Hugging Face `Dataset`, FiftyOne reads
    the Parquet and video files itself.
    """

    missing = [str(f) for f in _episode_files(meta, episodes) if not (meta.root / f).exists()]
    if not missing:
        return

    revision = meta.revision
    if is_valid_version(revision):
        revision = get_safe_version(meta.repo_id, revision)
    logging.info("Downloading %d file(s) for %s from the Hub", len(missing), meta.repo_id)
    if root_requested:
        snapshot_download(
            meta.repo_id, repo_type="dataset", revision=revision, local_dir=meta.root, allow_patterns=missing
        )
    else:
        meta.root = Path(
            snapshot_download(
                meta.repo_id,
                repo_type="dataset",
                revision=revision,
                cache_dir=HF_LEROBOT_HUB_CACHE,
                allow_patterns=missing,
            )
        )


def _check_video_codecs(meta: LeRobotDatasetMetadata) -> None:
    """Warn for camera streams whose codec the FiftyOne App doesn't decode natively."""

    for key in meta.camera_keys:
        feature = meta.features.get(key) or {}
        info = feature.get("info") or {}
        codec = info.get("video.codec")
        if codec is not None and codec not in _APP_NATIVE_CODECS:
            logging.warning(
                "FiftyOne: camera '%s' is encoded with '%s'. The FiftyOne App decodes %s natively; other "
                "codecs depend on your browser's WebCodecs support and may not play.",
                key,
                codec,
                " and ".join(sorted(_APP_NATIVE_CODECS)),
            )


def serve_fiftyone_dataset_playback(
    repo_id: str,
    episode_index: int | None,
    *,
    root: str | Path | None = None,
    host: str | None = None,
    port: int | None = None,
    remote: bool = False,
    all_episodes: bool = False,
    persistent: bool = True,
    dataset_name: str | None = None,
) -> None:
    """Open a LeRobot dataset episode in the FiftyOne App and block until interrupted.

    Resolves the dataset root from `meta/` alone ([`~lerobot.datasets.LeRobotDatasetMetadata`]),
    downloads any missing `data/` or `videos/` files for the requested episode(s), then builds a
    FiftyOne dataset from that root with `fo.Dataset.from_dir` (the `fiftyone.types.LeRobotDataset`
    importer), launches the App on it and waits for Ctrl-C. FiftyOne reads the Parquet and video
    files itself, so no `LeRobotDataset` is constructed and no frame table is loaded into memory.

    Args:
        repo_id (`str`):
            Hub repository id of the dataset (e.g. `lerobot/pusht`), also used for the default
            FiftyOne dataset name.
        episode_index (`int | None`):
            Episode to load. Ignored when `all_episodes` is `True`; required otherwise.
        root (`str | Path`, *optional*):
            Local dataset directory. When omitted, the dataset is looked up in (or downloaded to) the
            LeRobot cache, as with `LeRobotDataset`.
        host (`str`, *optional*):
            Address the App server binds to. `None` defers to FiftyOne's own config
            (`fo.config.default_app_address`).
        port (`int`, *optional*):
            Port the App server listens on. `None` defers to FiftyOne's own config
            (`fo.config.default_app_port`, 5151 unless `FIFTYONE_DEFAULT_APP_PORT` is set).
        remote (`bool`, *optional*, defaults to `False`):
            Run as a remote session: don't open a browser, print SSH port-forwarding instructions
            instead. Use this on a headless machine.
        all_episodes (`bool`, *optional*, defaults to `False`):
            Load every episode of the dataset instead of just `episode_index`.
        persistent (`bool`, *optional*, defaults to `True`):
            Keep the FiftyOne dataset in the database after exit so it can be reopened with
            `fo.load_dataset(name)` and any tags or saved views made in the App survive. Pass `False`
            for a throwaway session. Re-running with the same name replaces the dataset in place, so
            repeated runs don't accumulate.
        dataset_name (`str`, *optional*):
            Name for the FiftyOne dataset. Defaults to a sanitized `<repo_id>-episode-<i>` (or
            `<repo_id>` with `all_episodes`). An existing dataset with this name is replaced.

    Raises:
        ValueError: If `episode_index` is `None` and `all_episodes` is `False`.
    """

    require_package("fiftyone", extra="fiftyone")
    import fiftyone as fo
    import fiftyone.types as fot

    if not all_episodes and episode_index is None:
        raise ValueError("episode_index is required unless all_episodes=True.")

    episodes = None if all_episodes else [episode_index]
    name = dataset_name or _fiftyone_dataset_name(repo_id, None if all_episodes else episode_index)

    # Metadata only: this reads (or downloads) ``meta/`` and nothing else.
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    _ensure_files_on_disk(meta, episodes, root_requested=root is not None)
    # The importer resolves media relative to the dataset root and rejects relative paths.
    dataset_dir = str(meta.root.resolve())

    _check_video_codecs(meta)

    logging.info("Importing %s into FiftyOne dataset '%s'", dataset_dir, name)
    fo_dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fot.LeRobotDataset,
        episodes=episodes,
        name=name,
        persistent=persistent,
        overwrite=True,
    )

    # One FiftyOne sample per episode; check the import produced what was asked for.
    expected = meta.total_episodes if all_episodes else 1
    imported = len(fo_dataset)
    skipped = (fo_dataset.info.get("lerobot") or {}).get("skipped_episodes") or []
    if imported != expected or skipped:
        logging.warning(
            "FiftyOne: imported %d of %d episode(s); %d skipped%s.",
            imported,
            expected,
            len(skipped),
            f" ({'; '.join(map(str, skipped))})" if skipped else "",
        )

    session = fo.launch_app(fo_dataset, port=port, address=host, remote=remote)
    if not remote:
        print(f"FiftyOne App running at {session.url}")
    print("Ctrl-C to exit.")
    with contextlib.suppress(KeyboardInterrupt):
        session.wait(-1)
    # Leading newline: the terminal echoes ``^C`` on the current line, which would offset the box.
    print("\n" + _exit_message(name, persistent))
    # No explicit ``session.close()``: FiftyOne's documented script pattern is ``launch_app()`` +
    # ``wait()``, with the App shutting down when the process exits (``close()`` is for interactive
    # sessions that keep running afterwards, and can block on the server's open event stream here).


def _boxed(title: str, lines: list[str]) -> str:
    """Render ``lines`` inside a box-drawing frame whose top edge carries ``title``."""

    inner = max(len(line) for line in lines + [title]) + 2  # one space of padding each side
    top = f"┌─ {title} " + "─" * (inner - len(title) - 3) + "┐"
    body = [f"│ {line.ljust(inner - 2)} │" for line in lines]
    bottom = "└" + "─" * inner + "┘"
    return "\n".join([top, *body, bottom])


def _exit_message(name: str, persistent: bool) -> str:
    """What to print after the App is closed: how to get back to the FiftyOne dataset (or that it's gone)."""

    if not persistent:
        return (
            f"FiftyOne App closed. Dataset '{name}' was temporary; drop --no-persistent to keep tags "
            "and views between runs."
        )
    return _boxed(
        "FiftyOne",
        [
            f"Dataset saved: {name}",
            "",
            "Reopen from the terminal:",
            f"  fiftyone app launch {shlex.quote(name)}",
            "",
            "Reopen from Python:",
            "  import fiftyone as fo",
            f'  dataset = fo.load_dataset("{name}")',
            "  session = fo.launch_app(dataset)",
            "  session.wait()",
            "",
            "Learn more: https://docs.voxel51.com/user_guide/index.html",
        ],
    )
