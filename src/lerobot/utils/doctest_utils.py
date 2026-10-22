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

"""Doctest plumbing so the examples in our docstrings actually run.

Adapted from `transformers.testing_utils`. Two stdlib limitations make this necessary:

1. Ruff is configured with `docstring-code-format = true`, which reformats code inside docstrings and
   removes the blank line before the closing fence. stdlib's `_EXAMPLE_RE` then swallows the ` ``` ` into
   the expected-output group, so every example that has output fails. [`LeRobotDocTestParser`] patches the
   regex to stop at a fence.
2. `doctest.DocTestFinder` reports the wrong line number for `@property` and `functools.wraps` objects
   (https://bugs.python.org/issue17446). Our hardware API is property-heavy — `observation_features`,
   `action_features`, `is_connected`, `is_calibrated` are all abstract properties — so
   [`LeRobotDoctestModule`] unwraps them before locating the example.

Two environment variables skip whole example blocks by content:

- `SKIP_CUDA_DOCTEST=1` skips examples that need a GPU.
- `SKIP_HARDWARE_DOCTEST=1` skips examples that need a physical robot or a Hub download.

Both are heuristics over the example source. They are deliberately blunt: an example that is skipped
needlessly costs nothing, whereas one that runs on a machine without the hardware hangs or fails.
"""

import doctest
import functools
import inspect
import os
import re
import sys
from collections.abc import Iterable

from _pytest.doctest import (
    DoctestItem,
    DoctestModule,
    _get_checker,
    _get_continue_on_failure,
    _get_runner,
    get_optionflags,
)
from _pytest.nodes import Collector
from _pytest.outcomes import skip

# Calls whose progress bars would otherwise be compared against the expected output. The lookahead leaves
# lines that already carry a directive alone.
_NOISY_CALL_PATTERN = re.compile(r"(>>> (?!.*# doctest:).*(?:load_dataset|LeRobotDataset)\(.*)")

_CUDA_PATTERN = re.compile(r"cuda|to\(0\)|device=0")

# Serial ports, video devices, and the connect/scan calls that talk to real hardware.
_HARDWARE_PATTERN = re.compile(r"/dev/tty|/dev/video|COM\d|\.connect\(|find_cameras\(|find_port\(")

# Anything that reaches the Hub over the network.
_HUB_PATTERN = re.compile(r"from_pretrained\(|push_to_hub\(|snapshot_download\(|load_dataset\(")


def preprocess_string(string: str, skip_cuda_tests: bool, skip_hardware_tests: bool) -> str:
    """Prepare a docstring or `.mdx` file to be run by doctest.

    Args:
        string (`str`):
            A whole file's contents for `.mdx`, or a single docstring for a Python file. Either may hold
            several fenced examples.
        skip_cuda_tests (`bool`):
            Whether to drop examples that look like they need a GPU.
        skip_hardware_tests (`bool`):
            Whether to drop examples that look like they need a robot or a Hub download.

    Returns:
        `str`: The input with `# doctest: +IGNORE_RESULT` injected on noisy calls, or an empty string if
        the examples were skipped — in which case no doctest is collected for it at all.
    """
    # Match against the example lines only, not the surrounding prose, so that a docstring merely
    # *describing* CUDA or a serial port is not mistaken for one that uses them.
    example_lines = "\n".join(
        line for line in string.splitlines() if line.lstrip().startswith((">>>", "..."))
    )
    if not example_lines:
        return string

    if skip_cuda_tests and _CUDA_PATTERN.search(example_lines):
        return ""
    if skip_hardware_tests and (
        _HARDWARE_PATTERN.search(example_lines) or _HUB_PATTERN.search(example_lines)
    ):
        return ""

    return _NOISY_CALL_PATTERN.sub(r"\1 # doctest: +IGNORE_RESULT", string)


class LeRobotDocTestParser(doctest.DocTestParser):
    """A `DocTestParser` that understands fenced, auto-formatted code blocks.

    Ruff's `docstring-code-format` removes the blank line before a closing fence, after which stdlib's
    `_EXAMPLE_RE` reads the fence itself as part of the expected output and every example with output
    fails. The regex below is the stdlib one plus a clause that stops matching at a fence.
    """

    # fmt: off
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
             (?![ ]*>>>)          # Not a line starting with PS1
             (?:(?!```).)*        # Stop at a closing fence: formatting drops the blank line before it
             (?:\n|$)  # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE
    )
    # fmt: on

    skip_cuda_tests: bool = os.environ.get("SKIP_CUDA_DOCTEST", "0") == "1"
    skip_hardware_tests: bool = os.environ.get("SKIP_HARDWARE_DOCTEST", "0") == "1"

    def parse(self, string, name="<string>"):
        """Preprocess `string`, then parse it as stdlib would.

        Args:
            string (`str`):
                The docstring or file contents to parse.
            name (`str`, *optional*, defaults to `"<string>"`):
                Name used in failure messages.

        Returns:
            `list`: The examples and interleaved text, as returned by `doctest.DocTestParser.parse`.
        """
        string = preprocess_string(string, self.skip_cuda_tests, self.skip_hardware_tests)
        return super().parse(string, name)


class LeRobotDoctestModule(DoctestModule):
    """A pytest `DoctestModule` that collects with [`LeRobotDocTestParser`].

    `doctest.DocTestFinder` binds its default parser at class-definition time, so patching
    `doctest.DocTestParser` in `conftest.py` does not reach the finder pytest builds. The parser has to be
    passed in explicitly, which means reimplementing `collect`. It mirrors pytest's own implementation.
    """

    def collect(self) -> Iterable[DoctestItem]:
        """Collect the doctests in this module.

        Returns:
            `Iterable[DoctestItem]`: One item per example-bearing docstring. Docstrings whose examples were
            dropped by `preprocess_string` yield nothing.
        """

        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A doctest finder that reports correct line numbers for properties and wrapped callables."""

            # Fixed upstream in CPython 3.11.9 / 3.12.3; kept for older interpreters. Our hardware API is
            # property-heavy (`observation_features`, `is_connected`, ...), so a wrong line number here
            # would point every failure at the decorator. https://github.com/python/cpython/issues/61648
            def _find_lineno(self, obj, source_lines):
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)
                if hasattr(obj, "__wrapped__"):
                    obj = inspect.unwrap(obj)
                return super()._find_lineno(obj, source_lines)

            if sys.version_info < (3, 13):
                # `cached_property` is otherwise never considered part of the current module and its
                # examples are silently skipped. https://github.com/python/cpython/issues/107995
                def _from_module(self, module, object):
                    if isinstance(object, functools.cached_property):
                        object = object.func
                    return super()._from_module(module, object)

        try:
            module = self.obj
        except Collector.CollectError:
            if self.config.getvalue("doctest_ignore_import_errors"):
                skip(f"unable to import module {self.path!r}")
            else:
                raise

        # Doctests support fixtures via `getfixture` and autouse.
        self.session._fixturemanager.parsefactories(self)

        finder = MockAwareDocTestFinder(parser=LeRobotDocTestParser())
        optionflags = get_optionflags(self.config)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )
        for test in finder.find(module, module.__name__):
            if test.examples:  # Skip docstrings with no examples, and blocks dropped by the parser.
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
