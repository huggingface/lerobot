# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import dataclasses

import pytest

pytest.importorskip("wandb", reason="wandb is required (install lerobot[training])")

from lerobot.integrations.wandb_artifacts.refs import ArtifactRef, parse_artifact_ref


def test_parse_version_form():
    ref = parse_artifact_ref("my-team/my-project/pick-cube:v3")
    assert ref == ArtifactRef(entity="my-team", project="my-project", name="pick-cube", version_or_alias="v3")
    assert str(ref) == "my-team/my-project/pick-cube:v3"


def test_parse_alias_form():
    ref = parse_artifact_ref("my-team/my-project/pick-cube:latest")
    assert ref.version_or_alias == "latest"


def test_parse_allows_dots_and_underscores_in_components():
    ref = parse_artifact_ref("my.team/my_project/pick_cube.v2:candidate")
    assert ref.entity == "my.team"
    assert ref.project == "my_project"
    assert ref.name == "pick_cube.v2"


def test_ref_is_immutable():
    ref = parse_artifact_ref("a/b/c:v0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.entity = "other"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "a/b/c",  # missing version/alias
        "a/b:v0",  # missing project
        "a:v0",  # missing project and name
        "/b/c:v0",  # empty entity
        "a//c:v0",  # empty project
        "a/b/:v0",  # empty name
        "a/b/c:",  # empty version/alias
        "a/b/c: ",  # whitespace-only version/alias
        " a/b/c:v0",  # leading whitespace
        "a/b/c:v0 ",  # trailing whitespace
        "/home/user/datasets/pick-cube",  # local path
        "./relative/dataset",  # relative local path
        "wandb://a/b/c:v0",  # wandb:// style
        "a/b/c:v0/extra",  # too many components
    ],
)
def test_parse_rejects_malformed_refs(raw):
    with pytest.raises(ValueError):
        parse_artifact_ref(raw)


def test_parse_rejects_non_string_input():
    with pytest.raises(ValueError):
        parse_artifact_ref(None)  # type: ignore[arg-type]
