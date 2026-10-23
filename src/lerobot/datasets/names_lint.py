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
"""Non-blocking, machine-readable lint checks for LeRobot feature names.

Run ``python -m lerobot.datasets.names_lint DATASET_ROOT`` to write a JSON
report next to ``meta/info.json``. The report does not modify dataset metadata
and warnings do not affect dataset loading or upload.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Names such as ``motor_0`` are *positional placeholders*, not identities.
# This is a heuristic, not an ontology or a restriction on legal names.
_PLACEHOLDER_NAME = re.compile(r"(?:motor|joint|servo|actuator|axis|channel|dof)[_-]?\d+", re.IGNORECASE)
_VISUAL_DTYPES = frozenset({"image", "video"})
_REPORT_NAME = "names_lint.json"


def _flatten_names(names: Any) -> list[str] | None:
    """Flatten flat, nested, grouped and name-to-index representations.

    Return None for an unrecognized representation, without rejecting a dataset.
    Empty declarations are classified by the caller before flattening.
    """
    values: Iterable[Any]
    if isinstance(names, str):
        return [names]
    if isinstance(names, Mapping):
        if names and all(isinstance(value, int) and not isinstance(value, bool) for value in names.values()):
            indexed = sorted(names.items(), key=lambda pair: pair[1])
            if [index for _, index in indexed] != list(range(len(indexed))):
                return None
            if not all(isinstance(name, str) for name, _ in indexed):
                return None
            return [name for name, _ in indexed]
        values = names.values()
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes, bytearray)):
        values = names
    else:
        return None

    flattened: list[str] = []
    for value in values:
        nested = _flatten_names(value)
        if nested is None:
            return None
        flattened.extend(nested)
    return flattened


def _shape_size(shape: Any) -> int | None:
    """Return the number of scalar entries in a valid shape, if known."""
    if not isinstance(shape, (list, tuple)):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
        return None
    return math.prod(shape)


_BOOKKEEPING_FEATURES = frozenset({"timestamp", "frame_index", "episode_index", "index", "task_index"})
_NUMERIC_PREFIXES = ("float", "int", "uint", "bool")


def _is_empty_declaration(names: Any) -> bool:
    """Detect empty/group-null declarations without conflating them with absent names."""
    if names is None:
        return True
    if isinstance(names, Mapping):
        return not names or all(_is_empty_declaration(value) for value in names.values())
    if isinstance(names, (list, tuple)):
        return not names or all(_is_empty_declaration(value) for value in names)
    return False


def lint_feature_names(features: Mapping[str, Any]) -> dict[str, Any]:
    """Classify features and report actionable, non-blocking name warnings.

    The scalar-name count check applies to 1D numeric data. Image/video
    names describe axes, and higher-dimensional arrays may describe axes or
    only the last dimension, so these cases are classified but not linted.
    A singleton name on a larger 1D array may be a whole-vector label.

    Args:
        features (`Mapping[str, Any]`):
            Feature definitions from dataset metadata.

    Returns:
        `dict[str, Any]`: Per-feature classifications and structured warnings.
    """
    issues: list[dict[str, Any]] = []
    feature_classes: dict[str, str] = {}

    def warn(code: str, feature: str, message: str, **details: Any) -> None:
        issues.append({"code": code, "feature": feature, "message": message, "details": details})

    for feature_name, feature in sorted(features.items()):
        if not isinstance(feature, Mapping):
            feature_classes[feature_name] = "unsupported"
            warn("invalid_feature", feature_name, "Feature metadata is not a mapping.")
            continue

        dtype = feature.get("dtype")
        if feature_name in _BOOKKEEPING_FEATURES:
            feature_classes[feature_name] = "bookkeeping_excluded"
            continue
        if isinstance(dtype, str) and dtype in _VISUAL_DTYPES:
            feature_classes[feature_name] = "visual_excluded"
            continue
        if not isinstance(dtype, str) or not dtype.startswith(_NUMERIC_PREFIXES):
            feature_classes[feature_name] = "non_numeric_excluded"
            continue

        shape = feature.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 1 or _shape_size(shape) is None:
            feature_classes[feature_name] = "multidimensional_excluded"
            continue
        size = shape[0]
        raw_names = feature.get("names")
        if raw_names is None:
            feature_classes[feature_name] = "absent"
            continue
        if _is_empty_declaration(raw_names):
            feature_classes[feature_name] = "empty_declaration"
            warn("empty_declaration", feature_name, "Channel names are declared but empty.")
            continue

        names = _flatten_names(raw_names)
        if names is None:
            feature_classes[feature_name] = "unsupported"
            warn(
                "unsupported_names",
                feature_name,
                "Feature names are not a supported flat or grouped structure.",
            )
            continue
        if len(names) == 1 and size > 1:
            # e.g. LIBERO state=[8], names=["state"], or Metaworld ["keypoints"].
            # Do not classify these as a definite error: their granularity may be vector-wide.
            feature_classes[feature_name] = "possible_whole_vector"
            continue
        if len(names) != size:
            feature_classes[feature_name] = "mismatch"
            warn(
                "name_shape_mismatch",
                feature_name,
                "Number of channel names differs from the size of the 1D feature.",
                expected=size,
                actual=len(names),
                shape=list(shape),
            )
        else:
            feature_classes[feature_name] = "count_match"

        seen: dict[str, list[int]] = {}
        for index, name in enumerate(names):
            seen.setdefault(name, []).append(index)
        for name, positions in seen.items():
            if len(positions) > 1:
                warn(
                    "duplicate_name",
                    feature_name,
                    "One channel name occurs more than once in this feature.",
                    name=name,
                    positions=positions,
                )

        placeholders = [name for name in names if _PLACEHOLDER_NAME.fullmatch(name)]
        if placeholders:
            warn(
                "placeholder_names",
                feature_name,
                "Positional placeholder names do not attest channel identity.",
                names=placeholders,
            )

    return {
        "schema_version": 1,
        "issue_count": len(issues),
        "warnings_by_code": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        "feature_classes": feature_classes,
        "issues": issues,
    }


def lint_dataset_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Lint the features section of an info.json mapping.

    Args:
        info (`Mapping[str, Any]`):
            Parsed dataset metadata.

    Returns:
        `dict[str, Any]`: Per-feature classifications and structured warnings.
    """
    features = info.get("features", {})
    if not isinstance(features, Mapping):
        return {
            "schema_version": 1,
            "issue_count": 1,
            "warnings_by_code": {"invalid_features": 1},
            "feature_classes": {},
            "issues": [
                {
                    "code": "invalid_features",
                    "feature": "features",
                    "message": "The info.json features field is not a mapping.",
                    "details": {},
                }
            ],
        }
    return lint_feature_names(features)


def write_names_lint_report(dataset_root_or_info: str | Path, output: str | Path | None = None) -> Path:
    """Read info.json and write an actionable JSON report; never edit info.json.

    Args:
        dataset_root_or_info (`str | Path`):
            Dataset directory or metadata file.
        output (`str | Path`, *optional*):
            Report destination. Defaults to names_lint.json beside the metadata.

    Returns:
        `Path`: The written report path.

    Raises:
        FileNotFoundError: If the input metadata does not exist.
        ValueError: If the metadata is not an object or output aliases the input.
    """
    source = Path(dataset_root_or_info)
    info_path = source if source.is_file() else source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, Mapping):
        raise ValueError(f"Expected a JSON object in {info_path}")

    report = lint_dataset_info(info)
    destination = Path(output) if output is not None else info_path.with_name(_REPORT_NAME)
    if destination.resolve() == info_path.resolve() or (
        destination.exists() and destination.samefile(info_path)
    ):
        raise ValueError("The report output must not overwrite the input metadata.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    """Write a lint report from command-line arguments.

    Args:
        argv (`list[str]`, *optional*):
            Arguments to parse. Defaults to the process command line.

    Returns:
        `int`: Zero when a report was written, including reports with warnings.
    """
    parser = argparse.ArgumentParser(description="Write a non-blocking LeRobot feature-name lint report.")
    parser.add_argument("dataset", type=Path, help="Dataset root directory or meta/info.json")
    parser.add_argument("--output", type=Path, help="Report path (default: meta/names_lint.json)")
    args = parser.parse_args(argv)
    destination = write_names_lint_report(args.dataset, args.output)
    report = json.loads(destination.read_text(encoding="utf-8"))
    print(f"Found {report['issue_count']} feature-name warning(s); JSON report: {destination}")
    return 0  # Bad names never block loading or uploads.


if __name__ == "__main__":
    raise SystemExit(main())
