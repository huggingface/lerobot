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
import subprocess
import sys
import textwrap

import pytest

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
        "a/b/c",
        "a/b:v0",
        "a:v0",
        "/b/c:v0",
        "a//c:v0",
        "a/b/:v0",
        "a/b/c:",
        "a/b/c: ",
        " a/b/c:v0",
        "a/b/c:v0 ",
        "/home/user/datasets/pick-cube",
        "./relative/dataset",
        "wandb://a/b/c:v0",
        "a/b/c:v0/extra",
    ],
)
def test_parse_rejects_malformed_refs(raw):
    with pytest.raises(ValueError):
        parse_artifact_ref(raw)


def test_parse_rejects_non_string_input():
    with pytest.raises(ValueError):
        parse_artifact_ref(None)  # type: ignore[arg-type]


def test_reference_parser_imports_without_dataset_or_wandb():
    preamble = textwrap.dedent(
        """
        import builtins
        blocked = ("datasets", "wandb")
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == package or name.startswith(package + ".") for package in blocked):
                raise ModuleNotFoundError(name + " deliberately unavailable")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            preamble
            + textwrap.dedent(
                """
                from lerobot.integrations.wandb_artifacts.refs import ArtifactRef, parse_artifact_ref
                assert parse_artifact_ref("entity/project/name:v0") == ArtifactRef(
                    entity="entity",
                    project="project",
                    name="name",
                    version_or_alias="v0",
                )
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
