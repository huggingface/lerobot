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

import builtins
import io
import pickle
import shutil
from typing import Any

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("lerobot_lancedb")

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub.constants import ENDPOINT as HF_ENDPOINT
from lerobot_lancedb.convert import convert
from PIL import Image

from lerobot.datasets import lance_backend, lance_utils
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import DatasetImage
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils import import_utils
from tests.datasets import test_lance_backend as lance_fixtures
from tests.datasets.test_lance_backend import assert_items_equal
from tests.fixtures.constants import DUMMY_REPO_ID

pytestmark = pytest.mark.dataset

camera_dataset_roots = lance_fixtures.camera_dataset_roots
image_dataset_roots = lance_fixtures.image_dataset_roots
mixed_image_dataset_roots = lance_fixtures.mixed_image_dataset_roots
video_dataset_roots = lance_fixtures.video_dataset_roots


def assert_read_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, Image.Image):
        assert isinstance(actual, Image.Image)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    elif isinstance(expected, pa.Table):
        assert_read_equal(actual.to_pylist(), expected.to_pylist())
    elif isinstance(expected, pd.DataFrame):
        assert_read_equal(actual.to_dict("list"), expected.to_dict("list"))
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_read_equal(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            assert_read_equal(value, reference)
    else:
        assert actual == expected


def test_raw_rows_keep_base_semantics(camera_dataset_roots):
    src_root, lance_root = camera_dataset_roots
    kwargs = {
        "episodes": [2, 0],
        "delta_timestamps": {"observation.images.front": [-1 / 30, 0.0]},
        "depth_output_unit": "m",
        "image_transforms": lambda image: image + 0.2,
        "video_backend": "pyav",
    }
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, **kwargs)
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, **kwargs)
    for index in [0, 7, -1]:
        raw = lance_ds.get_raw_item(index)
        assert_read_equal(raw, upstream.get_raw_item(index))
        assert "task" not in raw
        assert not any(key.endswith("_is_pad") for key in raw)
        assert raw["observation.images.front"].ndim == 3
        assert raw["observation.images.depth_uint16"][0, 0, 0] >= 1000
        assert not set(lance_ds.meta.video_keys).intersection(raw)


@pytest.mark.parametrize("format_type", ["default", None, "numpy", "torch", "arrow", "pandas"])
def test_projected_formats_match_default(camera_dataset_roots, format_type):
    src_root, lance_root = camera_dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, episodes=[2, 0], video_backend="pyav")
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, episodes=[2, 0], video_backend="pyav")
    keys = ["action", "observation.images.front", "observation.images.depth_uint16"]
    expected = upstream.select_columns(keys)
    actual = lance_ds.select_columns(keys)
    assert len(actual) == len(expected)
    assert actual.column_names == expected.column_names
    assert actual.features == expected.features
    if format_type != "default":
        expected = expected.with_format(format_type)
        actual = actual.with_format(format_type)
    for query in [0, -1, slice(2, 11, 3), [7, 0, 7, 11], []]:
        assert_read_equal(actual[query], expected[query])


def test_projection_is_lazy_and_avoids_images(image_dataset_roots, monkeypatch):
    src_root, lance_root = image_dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root)
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)

    def fail_decode(*args, **kwargs):
        raise AssertionError("numeric projection decoded an image")

    monkeypatch.setattr(DatasetImage, "decode_example", fail_decode)
    view = lance_ds.select_columns(["action", "timestamp"])
    assert lance_ds.reader._frames_perm is None
    expected = upstream.select_columns(["action", "timestamp"])
    assert_read_equal(view[7], expected[7])
    assert_read_equal(list(view), list(expected))
    assert_read_equal(list(view["action"]), list(expected["action"]))
    assert_read_equal(view["action"][2:5], expected["action"][2:5])
    assert_read_equal(np.asarray(view.with_format("numpy")["timestamp"]), np.asarray(expected["timestamp"]))
    assert_read_equal(pickle.loads(pickle.dumps(view))[7], expected[7])
    assert all(set(columns) <= {"action", "timestamp"} for columns in lance_ds.reader._raw_perms)
    with pytest.raises(ValueError):
        view.select_columns("observation.images.front")
    with pytest.raises(ValueError):
        lance_ds.select_columns("missing")


@pytest.mark.parametrize("format_type", ["default", None, "numpy", "torch"])
@pytest.mark.parametrize("iteration", ["rows", "image_column", "numeric_column"])
def test_iteration_does_not_decode_unrequested_rows(
    image_dataset_roots, monkeypatch, format_type: str | None, iteration: str
) -> None:
    src_root, lance_root = image_dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root)
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)
    image_key = "observation.images.front"
    keys = ["action", *lance_ds.meta.image_keys]
    expected = upstream.select_columns(keys)
    view = lance_ds.select_columns(keys)
    if format_type != "default":
        expected = expected.with_format(format_type)
        view = view.with_format(format_type)
    calls: list[list[int]] = []
    read_rows = lance_ds.reader.read_rows

    def capture(indices: list[int], *args: Any, **kwargs: Any) -> Any:
        calls.append(list(indices))
        return read_rows(indices, *args, **kwargs)

    monkeypatch.setattr(lance_ds.reader, "read_rows", capture)
    key = image_key if iteration == "image_column" else "action"
    iterator = iter(view if iteration == "rows" else view[key])
    for index in range(2):
        reference = expected[index] if iteration == "rows" else expected[index][key]
        assert_read_equal(next(iterator), reference)
    if iteration == "numeric_column":
        assert calls == [list(range(min(512, len(view))))]
    else:
        assert calls == [[0], [1]]
    iterator.close()
    # Explicit batch iteration still honors the caller's requested size.
    calls.clear()
    assert_read_equal(next(view.iter(batch_size=3)), expected[:3])
    assert calls == [[0, 1, 2]]


def test_numeric_consumers_use_projected_rows(camera_dataset_roots, monkeypatch):
    from lerobot.policies.groot.processor_groot import _iter_action_state_training_samples
    from lerobot.scripts.augment_dataset_quantile_stats import collect_episode_arrays
    from lerobot.scripts.lerobot_train_tokenizer import process_episode

    src_root, lance_root = camera_dataset_roots
    kwargs = {
        "episodes": [2, 0],
        "delta_timestamps": {"action": [0.0, 1 / 30, 2 / 30], "observation.state": [-1 / 30, 0.0]},
        "video_backend": "pyav",
    }
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, **kwargs)
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, **kwargs)

    def fail_decode(*args, **kwargs):
        raise AssertionError("numeric consumer decoded an image")

    monkeypatch.setattr(DatasetImage, "decode_example", fail_decode)
    expected = list(_iter_action_state_training_samples(upstream))
    actual = list(_iter_action_state_training_samples(lance_ds))
    assert_read_equal(actual, expected)
    assert actual[0][1].shape == (2,)
    assert actual[-1][2].tolist() == [False, True, True]
    chunks = [
        process_episode((dataset, 2, 2, None, 1.0, "observation.state", False))
        for dataset in (upstream, lance_ds)
    ]
    assert all(isinstance(chunk, np.ndarray) and chunk.shape == (5, 2, 2) for chunk in chunks)
    assert_read_equal(chunks[1], chunks[0])
    aliased = [
        process_episode((dataset, 2, 2, None, 1.0, "action", False)) for dataset in (upstream, lance_ds)
    ]
    assert_read_equal(aliased[1], chunks[0])
    datasets = [
        LeRobotDataset(DUMMY_REPO_ID, root=root, video_backend="pyav") for root in (src_root, lance_root)
    ]
    expected_arrays = collect_episode_arrays(datasets[0], 0, skip_images=True)
    actual_arrays = collect_episode_arrays(datasets[1], 0, skip_images=True)
    assert_read_equal(actual_arrays, expected_arrays)


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "memory", "file_uri"])
def test_external_images_use_dataset_root(image_dataset_roots, tmp_path, monkeypatch, path_kind):
    original_src_root, _ = image_dataset_roots
    src_root = tmp_path / "src"
    shutil.copytree(original_src_root, src_root)
    asset = src_root / "assets" / "frame.png"
    asset.parent.mkdir()
    Image.new("RGB", (64, 64), (12, 34, 56)).save(asset)
    if path_kind == "relative":
        path = "assets/frame.png"
    elif path_kind == "absolute":
        path = str(asset)
    elif path_kind == "file_uri":
        path = asset.as_uri()
    else:
        path = f"memory://lerobot-image-tests/{tmp_path.name}/frame.png"
        fsspec.filesystem("memory").pipe(path, asset.read_bytes())
    data_path = src_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path)
    key = "observation.images.front"
    column = table.column(key)
    values = column.to_pylist()
    values[7] = {"bytes": None, "path": path}
    table = table.set_column(
        table.schema.get_field_index(key), table.schema.field(key), pa.array(values, type=column.type)
    )
    pq.write_table(table, data_path)
    lance_root = tmp_path / "lance"
    convert(lance_root, root=src_root)
    shutil.copytree(src_root / "assets", lance_root / "assets")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root)
    # URI roots may cache metadata elsewhere; image paths still use the real root.
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root.as_uri())
    assert_read_equal(lance_ds.get_raw_item(7), upstream.get_raw_item(7))
    assert_items_equal(lance_ds[7], upstream[7])
    assert_read_equal(pickle.loads(pickle.dumps(lance_ds))[7], upstream[7])


@pytest.mark.parametrize("path_kind", ["relative", "https"])
@pytest.mark.parametrize(
    "token,storage_options,cached_token,effective_token",
    [
        (None, {"token": "storage-token"}, None, "storage-token"),
        ("explicit-token", {"token": "storage-token"}, "cached-token", "storage-token"),
        (False, {"token": "storage-token"}, "cached-token", "storage-token"),
        ("explicit-token", {"token": False}, "cached-token", False),
        ("explicit-token", None, "cached-token", "explicit-token"),
        (None, None, "cached-token", "cached-token"),
        (False, None, "cached-token", False),
        (None, None, None, None),
        (None, {"token": "storage-token", "revision": "image-branch"}, None, "storage-token"),
    ],
    ids=[
        "storage_only",
        "storage_override",
        "storage_override_false",
        "storage_false",
        "explicit",
        "cached",
        "disabled",
        "anonymous",
        "revision_override",
    ],
)
def test_external_hf_images_use_table_revision(
    image_dataset_roots,
    monkeypatch,
    path_kind: str,
    token: str | bool | None,
    storage_options: dict[str, str | bool] | None,
    cached_token: str | None,
    effective_token: str | bool | None,
) -> None:
    _, local_root = image_dataset_roots
    root = f"hf://datasets/{DUMMY_REPO_ID}"
    revision = "image-snapshot"
    meta = LeRobotDatasetMetadata(DUMMY_REPO_ID, root=local_root, revision=revision)
    monkeypatch.setattr(lance_backend, "resolve_lance_root", lambda *args: (root, local_root))
    monkeypatch.setattr(lance_utils.huggingface_hub, "get_token", lambda: cached_token)
    monkeypatch.setattr("huggingface_hub.utils._headers.get_token", lambda: cached_token)
    connection_options = []
    original_connect = lance_backend._connect

    def connect(
        uri: str, options: dict | None, revision: str | None = None, token: str | bool | None = None
    ) -> Any:
        connection_options.append(lance_utils._storage_options(uri, options, revision, token))
        return original_connect(str(local_root), None, token=False)

    monkeypatch.setattr(lance_backend, "_connect", connect)
    reader = lance_backend.LanceDatasetReader(
        meta, root=root, revision=revision, token=token, storage_options=storage_options
    )
    reader.get_raw_item(0)
    assert connection_options[0].get("token", token) == effective_token
    effective_revision = (storage_options or {}).get("revision", revision)
    assert connection_options[0]["revision"] == effective_revision
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2)).save(encoded, format="PNG")
    if path_kind == "relative":
        path = "assets/frame.png"
        expected_path = f"{root}@{effective_revision}/{path}"
        expected_options = {"token": effective_token} if effective_token is not None else {}
    else:
        path = expected_path = f"{HF_ENDPOINT}/datasets/{DUMMY_REPO_ID}/resolve/{revision}/assets/frame.png"
        expected_options = (
            {"headers": {"Authorization": f"Bearer {effective_token}"}}
            if isinstance(effective_token, str)
            else {}
        )

    def open_image(path, mode, **options):
        assert path == expected_path
        assert mode == "rb"
        assert options == expected_options
        return io.BytesIO(encoded.getvalue())

    monkeypatch.setattr(fsspec, "open", open_image)
    image = reader._decode_image({"bytes": None, "path": path}, meta.image_keys[0], 0)
    assert image.shape == (3, 2, 2)
    reader.close()


@pytest.mark.parametrize("root_suffix", ["/nested", "@uri-snapshot", "@uri-snapshot/nested"])
def test_relative_hf_images_keep_root_path_and_pinned_revision(
    image_dataset_roots, monkeypatch, root_suffix: str
) -> None:
    _, local_root = image_dataset_roots
    root = f"hf://datasets/{DUMMY_REPO_ID}{root_suffix}"
    meta = LeRobotDatasetMetadata(DUMMY_REPO_ID, root=local_root, revision="meta-snapshot")
    monkeypatch.setattr(lance_backend, "resolve_lance_root", lambda *args: (root, local_root))
    reader = lance_backend.LanceDatasetReader(
        meta, root=root, storage_options={"revision": "image-branch"}, token=False
    )
    expected_root = (
        root if root_suffix.startswith("@") else f"hf://datasets/{DUMMY_REPO_ID}@image-branch{root_suffix}"
    )
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2)).save(encoded, format="PNG")

    def open_image(path, mode, **options):
        assert path == f"{expected_root}/assets/frame.png"
        assert mode == "rb"
        assert options == {"token": False}
        return io.BytesIO(encoded.getvalue())

    monkeypatch.setattr(fsspec, "open", open_image)
    image = reader._decode_image({"bytes": None, "path": "assets/frame.png"}, meta.image_keys[0], 0)
    assert image.shape == (3, 2, 2)
    reader.close()


@pytest.mark.parametrize(
    "root,options",
    [
        ("hf://datasets/test/images@revision", {"token": "test-token"}),
        (
            f"{HF_ENDPOINT}/datasets/test/images/resolve/main",
            {"headers": {"Authorization": "Bearer test-token"}},
        ),
        (
            "s3://test/images",
            {
                "key": "test-key",
                "secret": "test-secret",
                "client_kwargs": {"endpoint_url": "https://test.invalid"},
            },
        ),
    ],
)
def test_external_image_credentials(root, options, monkeypatch):
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2)).save(encoded, format="PNG")
    calls = []

    def open_image(path, mode, **kwargs):
        calls.append((path, mode, kwargs))
        return io.BytesIO(encoded.getvalue())

    monkeypatch.setattr(fsspec, "open", open_image)
    feature = DatasetImage(
        root,
        token="test-token",
        storage_options={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "aws_endpoint": "https://test.invalid",
        },
    )
    with feature.decode_example({"bytes": None, "path": "frame.png"}) as image:
        assert image.size == (2, 2)
    assert calls == [(f"{root}/frame.png", "rb", options)]


@pytest.mark.parametrize("return_uint8", [False, True])
def test_pyav_video_reads_match_default(video_dataset_roots, return_uint8):
    src_root, lance_root = video_dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, video_backend="pyav", return_uint8=return_uint8)
    timestamps = {key: [-1 / upstream.fps, 0.0, 1 / upstream.fps] for key in upstream.meta.video_keys}
    kwargs = {"video_backend": "pyav", "return_uint8": return_uint8, "delta_timestamps": timestamps}
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, **kwargs)
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, **kwargs)
    indices = [0, 1, 1, 19, 20, 39]
    for index, actual in zip(indices, lance_ds.__getitems__(indices), strict=True):
        assert_items_equal(actual, upstream[index])


def test_pyav_and_auto_fallback_do_not_require_torchcodec(mixed_image_dataset_roots, monkeypatch):
    src_root, lance_root = mixed_image_dataset_roots
    expected = LeRobotDataset(DUMMY_REPO_ID, root=src_root, video_backend="pyav")[7]
    original_import = builtins.__import__
    import_module = import_utils.importlib.import_module

    def without_torchcodec(name, *args, **kwargs):
        if name.startswith("torchcodec"):
            raise ImportError("torchcodec unavailable")
        return original_import(name, *args, **kwargs)

    def without_torchcodec_module(name, *args, **kwargs):
        if name.startswith("torchcodec"):
            raise ImportError("torchcodec unavailable")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torchcodec)
    monkeypatch.setattr(import_utils.importlib, "import_module", without_torchcodec_module)
    for backend in ["pyav", None]:
        dataset = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, video_backend=backend)
        assert dataset.reader._video_backend == "pyav"
        assert_items_equal(dataset[7], expected)
    with pytest.raises(ImportError, match="video_backend='pyav'"):
        LeRobotDataset(DUMMY_REPO_ID, root=lance_root, video_backend="torchcodec")
