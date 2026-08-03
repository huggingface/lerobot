#!/usr/bin/env python

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402

from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.datasets.io_utils import write_info  # noqa: E402
from lerobot.datasets.language import (  # noqa: E402
    EVENT_ONLY_STYLES,
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    PERSISTENT_STYLES,
    STYLE_REGISTRY,
    VIEW_DEPENDENT_STYLES,
    column_for_style,
    is_view_dependent_style,
    language_events_arrow_type,
    language_feature_info,
    language_persistent_arrow_type,
    validate_camera_field,
)
from lerobot.datasets.utils import DEFAULT_DATA_PATH  # noqa: E402


def test_language_arrow_schema_has_expected_fields():
    persistent_row_type = language_persistent_arrow_type().value_type
    event_row_type = language_events_arrow_type().value_type

    assert isinstance(persistent_row_type, pa.StructType)
    assert persistent_row_type.names == [
        "role",
        "content",
        "style",
        "timestamp",
        "camera",
        "tool_calls",
    ]

    assert isinstance(event_row_type, pa.StructType)
    assert event_row_type.names == ["role", "content", "style", "camera", "tool_calls"]

    # Persistent-row timestamps use float32, matching LeRobotDataset frame timestamps.
    assert persistent_row_type.field("timestamp").type == pa.float32()


def test_validate_feature_language_warns_only_on_non_empty_value(caplog):
    from lerobot.datasets.feature_utils import validate_feature_language

    # None (the expected record-time value) is silent and non-fatal.
    with caplog.at_level("WARNING"):
        assert validate_feature_language("language_persistent", None) == ""
    assert caplog.records == []

    # A stray non-empty value is dropped later, so we warn rather than fail.
    with caplog.at_level("WARNING"):
        assert validate_feature_language("language_persistent", [{"role": "user"}]) == ""
    assert any("language_persistent" in r.message for r in caplog.records)


def test_style_registry_routes_columns():
    assert {"subtask", "plan", "memory", "motion", "task_aug"} == PERSISTENT_STYLES
    assert {"interjection", "vqa", "trace"} == EVENT_ONLY_STYLES
    assert PERSISTENT_STYLES | EVENT_ONLY_STYLES <= STYLE_REGISTRY

    assert column_for_style("subtask") == LANGUAGE_PERSISTENT
    assert column_for_style("plan") == LANGUAGE_PERSISTENT
    assert column_for_style("memory") == LANGUAGE_PERSISTENT
    assert column_for_style("motion") == LANGUAGE_PERSISTENT
    assert column_for_style("task_aug") == LANGUAGE_PERSISTENT
    assert column_for_style("interjection") == LANGUAGE_EVENTS
    assert column_for_style("vqa") == LANGUAGE_EVENTS
    assert column_for_style("trace") == LANGUAGE_EVENTS
    assert column_for_style(None) == LANGUAGE_EVENTS


def test_view_dependent_styles():
    # motion lives in PERSISTENT_STYLES and is described in robot-frame
    # (joint / Cartesian) terms, so it is NOT view-dependent. Only vqa
    # (event) and trace (event, pixel-trajectory) carry a camera tag.
    assert {"vqa", "trace"} == VIEW_DEPENDENT_STYLES
    assert is_view_dependent_style("vqa")
    assert is_view_dependent_style("trace")
    assert not is_view_dependent_style("motion")
    assert not is_view_dependent_style("subtask")
    assert not is_view_dependent_style("plan")
    assert not is_view_dependent_style("interjection")
    assert not is_view_dependent_style(None)


def test_validate_camera_field_requires_camera_for_view_dependent_styles():
    validate_camera_field("vqa", "observation.images.top")
    validate_camera_field("trace", "observation.images.front")
    with pytest.raises(ValueError, match="view-dependent"):
        validate_camera_field("vqa", None)
    with pytest.raises(ValueError, match="view-dependent"):
        validate_camera_field("trace", "")


def test_validate_camera_field_rejects_camera_on_non_view_dependent_styles():
    validate_camera_field("subtask", None)
    validate_camera_field("plan", None)
    validate_camera_field("memory", None)
    validate_camera_field("motion", None)
    validate_camera_field("interjection", None)
    validate_camera_field(None, None)
    with pytest.raises(ValueError, match="must have camera=None"):
        validate_camera_field("subtask", "observation.images.top")
    with pytest.raises(ValueError, match="must have camera=None"):
        validate_camera_field("motion", "observation.images.top")
    with pytest.raises(ValueError, match="must have camera=None"):
        validate_camera_field("interjection", "observation.images.top")
    with pytest.raises(ValueError, match="must have camera=None"):
        validate_camera_field(None, "observation.images.top")


def test_unknown_style_rejected():
    with pytest.raises(ValueError, match="Unknown language style"):
        column_for_style("surprise")


def test_lerobot_dataset_passes_language_columns_through(tmp_path, empty_lerobot_dataset_factory):
    root = tmp_path / "language_dataset"
    dataset = empty_lerobot_dataset_factory(
        root=root,
        features={"state": {"dtype": "float32", "shape": (2,), "names": None}},
        use_videos=False,
    )
    dataset.add_frame({"state": np.array([0.0, 1.0], dtype=np.float32), "task": "tidy"})
    dataset.add_frame({"state": np.array([1.0, 2.0], dtype=np.float32), "task": "tidy"})
    dataset.save_episode()
    dataset.finalize()

    persistent = [
        {
            "role": "assistant",
            "content": "reach for the cup",
            "style": "subtask",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        }
    ]
    event = {
        "role": "user",
        "content": "what is visible?",
        "style": "vqa",
        "camera": "observation.images.top",
        "tool_calls": None,
    }
    data_path = root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    df = pd.read_parquet(data_path)
    df[LANGUAGE_PERSISTENT] = [persistent, persistent]
    df[LANGUAGE_EVENTS] = [[event], []]
    df.to_parquet(data_path)

    info = dataset.meta.info
    info["features"].update(language_feature_info())
    write_info(info, root)

    reloaded = LeRobotDataset(repo_id=dataset.repo_id, root=root)

    first = reloaded[0]
    second = reloaded[1]
    assert first[LANGUAGE_PERSISTENT] == persistent
    assert first[LANGUAGE_EVENTS] == [event]
    assert second[LANGUAGE_PERSISTENT] == persistent
    assert second[LANGUAGE_EVENTS] == []
