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

import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT_PATH = Path(__file__).parents[1] / "pyproject.toml"
UV_LOCK_PATH = Path(__file__).parents[1] / "uv.lock"

CORE_TORCH_MINORS = tuple(Version(f"2.{minor}") for minor in range(7, 12))
CORE_TORCHVISION_MINORS = tuple(Version(f"0.{minor}") for minor in range(22, 27))
TORCHCODEC_ABI_PROBE_MINORS = tuple(Version(f"0.{minor}") for minor in range(10, 13))
LEROBOT_TORCH_MINOR = Version("2.11")
LEROBOT_TORCHVISION_MINOR = Version("0.26")
LEROBOT_TORCHCODEC_MINOR = Version("0.11")
TORCHCODEC_WHEEL_PLATFORMS = {
    ("darwin", "arm64"),
    ("linux", "AMD64"),
    ("linux", "aarch64"),
    ("linux", "arm64"),
    ("linux", "x86_64"),
    ("win32", "AMD64"),
    ("win32", "x86_64"),
}
CHECKED_PLATFORMS = TORCHCODEC_WHEEL_PLATFORMS | {
    ("darwin", "x86_64"),
    ("linux", "armv7l"),
    ("win32", "ARM64"),
}


def _minor(version: Version) -> Version:
    return Version(".".join(str(part) for part in version.release[:2]))


def _requirement_applies(requirement: Requirement, platform: tuple[str, str]) -> bool:
    if requirement.marker is None:
        return True
    environment = default_environment()
    environment.update(sys_platform=platform[0], platform_machine=platform[1])
    return requirement.marker.evaluate(environment)


def _allowed_versions(
    requirements: list[Requirement], candidates: tuple[Version, ...]
) -> tuple[Version, ...]:
    return tuple(version for version in candidates if all(version in req.specifier for req in requirements))


@pytest.mark.parametrize(
    "platform",
    sorted(CHECKED_PLATFORMS),
    ids=lambda platform: f"{platform[0]}-{platform[1]}",
)
def test_dataset_extra_scopes_its_torchcodec_abi_tuple_to_wheel_platforms(platform):
    with PYPROJECT_PATH.open("rb") as file:
        project = tomllib.load(file)["project"]

    base_requirements = [Requirement(value) for value in project["dependencies"]]
    dataset_requirements = [Requirement(value) for value in project["optional-dependencies"]["dataset"]]

    base_platform_requirements = [
        requirement for requirement in base_requirements if _requirement_applies(requirement, platform)
    ]
    dataset_platform_requirements = [
        requirement for requirement in dataset_requirements if _requirement_applies(requirement, platform)
    ]
    base_torch_requirements = [req for req in base_platform_requirements if req.name == "torch"]
    base_torchvision_requirements = [req for req in base_platform_requirements if req.name == "torchvision"]
    dataset_torch_requirements = [req for req in dataset_platform_requirements if req.name == "torch"]
    dataset_torchvision_requirements = [
        req for req in dataset_platform_requirements if req.name == "torchvision"
    ]
    torchcodec_requirements = [req for req in dataset_platform_requirements if req.name == "torchcodec"]

    has_torchcodec_wheel = platform in TORCHCODEC_WHEEL_PLATFORMS
    assert bool(dataset_torch_requirements) == has_torchcodec_wheel
    assert bool(dataset_torchvision_requirements) == has_torchcodec_wheel
    assert bool(torchcodec_requirements) == has_torchcodec_wheel

    effective_torch_requirements = base_torch_requirements + dataset_torch_requirements
    effective_torchvision_requirements = base_torchvision_requirements + dataset_torchvision_requirements
    allowed_torch = _allowed_versions(effective_torch_requirements, CORE_TORCH_MINORS)
    allowed_torchvision = _allowed_versions(effective_torchvision_requirements, CORE_TORCHVISION_MINORS)
    if has_torchcodec_wheel:
        assert allowed_torch == (LEROBOT_TORCH_MINOR,)
        assert allowed_torchvision == (LEROBOT_TORCHVISION_MINOR,)
        assert _allowed_versions(torchcodec_requirements, TORCHCODEC_ABI_PROBE_MINORS) == (
            LEROBOT_TORCHCODEC_MINOR,
        )
    else:
        assert allowed_torch == CORE_TORCH_MINORS
        assert allowed_torchvision == CORE_TORCHVISION_MINORS


def test_dataset_extra_keeps_torch_torchvision_and_torchcodec_platform_markers_in_sync():
    with PYPROJECT_PATH.open("rb") as file:
        dataset_requirements = [
            Requirement(value) for value in tomllib.load(file)["project"]["optional-dependencies"]["dataset"]
        ]

    dataset_torch_requirements = [req for req in dataset_requirements if req.name == "torch"]
    dataset_torchvision_requirements = [req for req in dataset_requirements if req.name == "torchvision"]
    torchcodec_requirements = [req for req in dataset_requirements if req.name == "torchcodec"]

    assert len(dataset_torch_requirements) == len(dataset_torchvision_requirements) == 1
    assert len(torchcodec_requirements) == 1
    # The ABI-matched tuple must activate on exactly the same platforms, including ones not sampled above.
    markers = {
        str(requirements[0].marker)
        for requirements in (
            dataset_torch_requirements,
            dataset_torchvision_requirements,
            torchcodec_requirements,
        )
    }
    assert len(markers) == 1


def test_locked_torch_torchvision_and_torchcodec_versions_match():
    with UV_LOCK_PATH.open("rb") as file:
        packages = tomllib.load(file)["package"]

    torch_versions = {Version(package["version"]) for package in packages if package["name"] == "torch"}
    torchvision_versions = {
        Version(package["version"]) for package in packages if package["name"] == "torchvision"
    }
    torchcodec_versions = {
        Version(package["version"]) for package in packages if package["name"] == "torchcodec"
    }
    assert {_minor(version) for version in torch_versions} == {LEROBOT_TORCH_MINOR}
    assert {_minor(version) for version in torchvision_versions} == {LEROBOT_TORCHVISION_MINOR}
    assert {_minor(version) for version in torchcodec_versions} == {LEROBOT_TORCHCODEC_MINOR}
