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
"""Image decoding context for the Lance dataset reader."""

from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import datasets
import fsspec
import torch
from huggingface_hub.constants import ENDPOINT as HF_ENDPOINT
from huggingface_hub.utils import get_token_to_send
from PIL import Image as PILImage

from lerobot.streaming.location import StorageLocation


class LanceImage(datasets.Image):
    """HF Image decoding with a dataset root and external-resource credentials.

    Root and credentials are runtime attributes, outside the inherited Image
    dataclass fields, so the persisted HF schema remains the standard Image
    schema. The context is retained by copies and worker pickling.
    """

    def __init__(
        self,
        root: str | Path,
        token: str | bool | None = None,
        storage_options: dict | None = None,
    ) -> None:
        super().__init__()
        self._root = str(root)
        self._token = token
        self._storage_options = dict(storage_options or {})

    def __eq__(self, other: object) -> bool:
        """Compare the persisted feature definition, excluding runtime context."""
        return isinstance(other, datasets.Image) and asdict(self) == asdict(other)

    def decode_example(
        self, value: dict, token_per_repo_id: dict[str, str | bool | None] | None = None
    ) -> PILImage.Image:
        """Decode embedded bytes or an accessible local/remote image path."""
        if value.get("bytes") is not None:
            return super().decode_example({"bytes": value["bytes"], "path": None}, token_per_repo_id)
        path = value.get("path")
        if path is None:
            raise ValueError("An image must contain either bytes or a path.")
        if "://" not in path and not Path(path).is_absolute():
            path = (
                f"{self._root.rstrip('/')}/{Path(path).as_posix()}"
                if "://" in self._root
                else str(Path(self._root).expanduser() / path)
            )
        options: dict = StorageLocation.parse(path).storage_options(self._token)
        parsed_path, hub_endpoint = urlsplit(path), urlsplit(HF_ENDPOINT)
        if parsed_path.scheme in {"http", "https"} and (parsed_path.scheme, parsed_path.netloc) == (
            hub_endpoint.scheme,
            hub_endpoint.netloc,
        ):
            access_token = get_token_to_send(self._token)
            if access_token is not None:
                options["headers"] = {"Authorization": f"Bearer {access_token}"}
        if path.startswith("s3://"):
            # Lance/object_store names its S3 credentials differently from fsspec.
            for source, target in (
                ("aws_access_key_id", "key"),
                ("aws_secret_access_key", "secret"),
                ("aws_session_token", "token"),
            ):
                if source in self._storage_options:
                    options[target] = self._storage_options[source]
            client = {
                target: self._storage_options[source]
                for source, target in (("aws_endpoint", "endpoint_url"), ("aws_region", "region_name"))
                if source in self._storage_options
            }
            if client:
                options["client_kwargs"] = client
        with fsspec.open(path, "rb", **options) as file:
            data = file.read()
        return super().decode_example({"bytes": data, "path": None}, token_per_repo_id)


def require_image(image: torch.Tensor | None, key: str, frame_index: int) -> torch.Tensor:
    """Require an image when assembling a training sample, with a useful error."""
    if image is None:
        raise ValueError(f"Missing image for feature {key!r} at frame {frame_index}.")
    return image
