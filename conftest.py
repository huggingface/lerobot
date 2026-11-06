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

"""Root conftest: makes doctest collection use LeRobot's parser.

This only affects `--doctest-modules` runs (see `make doctest`). The test suite itself is configured by
`tests/conftest.py`.
"""

import doctest

import _pytest.doctest

from lerobot.utils.doctest_utils import LeRobotDoctestModule, LeRobotDocTestParser

# Lets an example opt out of output comparison with `# doctest: +IGNORE_RESULT`, for calls whose output is
# a progress bar or otherwise not reproducible.
IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    """An output checker that honours the `IGNORE_RESULT` flag."""

    def check_output(self, want, got, optionflags):
        """Return `True` when `IGNORE_RESULT` is set, otherwise defer to stdlib.

        Args:
            want (`str`):
                The expected output.
            got (`str`):
                The actual output.
            optionflags (`int`):
                Bitmask of active doctest option flags.

        Returns:
            `bool`: Whether the output is considered a match.
        """
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


# Reassigning these module attributes is how doctest behaviour is customised; mypy sees it as assigning to
# a type, which is exactly what is intended here.
doctest.OutputChecker = CustomOutputChecker  # type: ignore[misc]
_pytest.doctest.DoctestModule = LeRobotDoctestModule
doctest.DocTestParser = LeRobotDocTestParser  # type: ignore[misc]
