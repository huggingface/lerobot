#!/bin/bash
#
# Copyright 2025 Enactic, Inc.
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

set -eu

if [ $# -ne 1 ]; then
    echo "Usage: $0 version"
    echo " e.g.: $0 1.0.0"
    exit 0
fi

version="$1"

base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${base_dir}"

if [ "${RELEASE_CHECK_ORIGIN:-yes}" = "yes" ]; then
    git_origin_url="$(git remote get-url origin)"
    if [ "${git_origin_url}" != "git@github.com:enactic/openarm_description.git" ]; then
        echo "This script must be ran with working copy of enactic/openarm_description."
        echo "The origin's URL: ${git_origin_url}"
        exit 1
    fi
fi

if [ "${RELEASE_PULL:-yes}" = "yes" ]; then
    echo "Ensure using the latest commit"
    git checkout main
    git pull --ff-only
fi

if [ "${RELEASE_TAG:-yes}" = "yes" ]; then
    echo "Tag"
    git tag -a -m "OpenArm Description ${version}" "${version}"
    git push origin "${version}"
fi
