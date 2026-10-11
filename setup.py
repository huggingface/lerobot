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


from setuptools import setup


def get_version_from_toml() -> str:
    """Return the project's version string parsed from `pyproject.toml`.

    The function scans `pyproject.toml` line-by-line looking for a line
    that starts with ``version`` (for example: ``version = "1.2.3"``)
    and returns the value without surrounding quotes. If no such line is
    found a :class:`ValueError` is raised.

    Returns:
        The version string from `pyproject.toml` (e.g. ``"1.2.3"`` ->
        ``1.2.3``).
    """

    version = None
    with open("pyproject.toml", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("version"):
                version = line.split("=")[1].strip().strip('"')
                break
    if version is None:
        raise ValueError("Version not found in pyproject.toml")
    return version


def read_long_description() -> str:
    """Read and return the project's long description for setup.

    This function reads `README.md` and replaces image links that point
    to the local `./media/` directory with absolute raw GitHub URLs that
    reference the release tag corresponding to the version parsed from
    `pyproject.toml` (for example, ``v1.2.3``). The modified README
    content is returned as a string suitable for passing to
    ``setuptools.setup(long_description=...)``.

    Returns:
        The README content with rewritten media links.
    """

    with open("README.md", encoding="utf-8") as f:
        content = f.read()

    version = get_version_from_toml()
    git_tag = f"v{version}"

    base_raw_url = f"https://raw.githubusercontent.com/huggingface/lerobot/{git_tag}/"
    content = content.replace('src="./media/', f'src="{base_raw_url}media/')

    return content


setup(
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
)
