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

from pathlib import Path

import pytest
from PIL import Image as PILImage

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, List, Value

from lerobot.datasets.io_utils import embed_images

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("indices", [[], [2, 0]])
@pytest.mark.parametrize("format_type", [None, "numpy", "torch"])
def test_embed_images_reuses_dataset_without_media(indices: list[int], format_type: str | None) -> None:
    dataset = Dataset.from_dict(
        {"state": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], "task": ["a", "b", "c"]},
        features=Features({"state": List(Value("float32")), "task": Value("string")}),
    )
    dataset = dataset.select(indices).with_format(format_type, columns=["state"], output_all_columns=True)

    # No storage needs embedding: preserve the table, selection and format without copying.
    assert embed_images(dataset) is dataset


@pytest.mark.parametrize("layout", ["flat", "struct", "list"])
def test_embed_images_keeps_external_images_in_parquet(tmp_path: Path, layout: str) -> None:
    image_path = tmp_path / "image.png"
    PILImage.new("RGB", (4, 3), color=(12, 34, 56)).save(image_path)
    expected_bytes = image_path.read_bytes()
    feature = Image(decode=False)
    value = str(image_path)
    if layout == "struct":
        feature = {"camera": feature}
        value = {"camera": value}
    elif layout == "list":
        feature = List(feature)
        value = [value]
    dataset = Dataset.from_dict(
        {"observation": [value], "index": [7]},
        features=Features({"observation": feature, "index": Value("int64")}),
    ).with_format("numpy", columns=["index"], output_all_columns=True)
    original_format = dataset.format

    embedded = embed_images(dataset)

    assert embedded.features == dataset.features
    assert embedded.format == original_format
    assert dataset.format == original_format
    # Use the same Arrow writer as dataset saving, which does not embed images itself.
    parquet_path = tmp_path / "data.parquet"
    pq.write_table(embedded.with_format("arrow")[:], parquet_path)
    image_path.unlink()
    row = pq.read_table(parquet_path).to_pylist()[0]
    image = row["observation"]
    if layout == "struct":
        image = image["camera"]
    elif layout == "list":
        image = image[0]
    assert row["index"] == 7
    assert image["bytes"] == expected_bytes
