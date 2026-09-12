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

import doctest

from lerobot.utils.doctest_utils import LeRobotDocTestParser, preprocess_string

# An example with expected output, formatted the way ruff's `docstring-code-format` leaves it: no blank
# line between the last output line and the closing fence. This is the exact shape that breaks stdlib.
FORMATTED_EXAMPLE = """Summary.

Example:
    ```python
    >>> 1 + 1
    2
    ```
"""


def test_stdlib_parser_swallows_the_closing_fence():
    """Guards the premise of the port: without the patch, the fence lands in the expected output.

    Uses the base class rather than `doctest.DocTestParser`, which the root `conftest.py` has already
    replaced with ours by the time this runs.
    """
    stdlib_parser = LeRobotDocTestParser.__bases__[0]()
    (example,) = (e for e in stdlib_parser.parse(FORMATTED_EXAMPLE) if isinstance(e, doctest.Example))
    assert "```" in example.want


def test_parser_stops_at_the_closing_fence():
    """The whole reason `LeRobotDocTestParser` exists: `want` must be the output and nothing else."""
    (example,) = (
        e for e in LeRobotDocTestParser().parse(FORMATTED_EXAMPLE) if isinstance(e, doctest.Example)
    )
    assert example.source == "1 + 1\n"
    assert example.want == "2\n"


def test_example_with_output_passes_end_to_end():
    """A formatted example with output should actually run green."""
    runner = doctest.DocTestRunner()
    test = LeRobotDocTestParser().get_doctest(FORMATTED_EXAMPLE, {}, "formatted", None, 0)
    results = runner.run(test, out=lambda _: None)
    assert results.failed == 0
    assert results.attempted == 1


def test_noisy_calls_get_ignore_result():
    string = """
    ```python
    >>> ds = load_dataset("lerobot/pusht")
    ```
    """
    assert "# doctest: +IGNORE_RESULT" in preprocess_string(string, False, False)


def test_ignore_result_is_not_added_twice():
    string = """
    ```python
    >>> ds = load_dataset("lerobot/pusht")  # doctest: +IGNORE_RESULT
    ```
    """
    assert preprocess_string(string, False, False).count("# doctest: +IGNORE_RESULT") == 1


def test_cuda_examples_are_dropped_when_requested():
    string = """
    ```python
    >>> model.to("cuda")
    ```
    """
    assert preprocess_string(string, True, False) == ""
    assert preprocess_string(string, False, False) != ""


def test_hardware_examples_are_dropped_when_requested():
    """Serial ports, connect calls and Hub downloads all need real resources."""
    for source in [
        '>>> robot = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM0"))',
        ">>> robot.connect()",
        '>>> policy = ACTPolicy.from_pretrained("lerobot/act")',
    ]:
        string = f"""
    ```python
    {source}
    ```
    """
        assert preprocess_string(string, False, True) == "", source
        assert preprocess_string(string, False, False) != "", source


def test_plain_examples_survive_both_skips():
    string = """
    ```python
    >>> 1 + 1
    2
    ```
    """
    assert preprocess_string(string, True, True) == string
