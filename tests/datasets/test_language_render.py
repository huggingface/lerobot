#!/usr/bin/env python

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.recipe import MessageTurn, TrainingRecipe  # noqa: E402
from lerobot.datasets.language_render import (  # noqa: E402
    EMITTED_AT_TOLERANCE_S,
    active_at,
    emitted_at,
    nth_next,
    nth_prev,
    render_sample,
)


def persistent_row(role, content, style, timestamp, tool_calls=None, camera=None):
    return {
        "role": role,
        "content": content,
        "style": style,
        "timestamp": timestamp,
        "camera": camera,
        "tool_calls": tool_calls,
    }


def event_row(role, content, style, tool_calls=None, camera=None):
    return {
        "role": role,
        "content": content,
        "style": style,
        "camera": camera,
        "tool_calls": tool_calls,
    }


PERSISTENT = [
    persistent_row("assistant", "plan 0", "plan", 0.0),
    persistent_row("assistant", "memory 0", "memory", 0.0),
    persistent_row("assistant", "subtask 0", "subtask", 0.0),
    persistent_row("assistant", "memory 1", "memory", 1.0),
    persistent_row("assistant", "subtask 1", "subtask", 1.0),
]
EVENTS_AT_1 = [
    event_row("user", "what is visible?", "vqa", camera="observation.images.top"),
    event_row("assistant", '{"count": 2}', "vqa", camera="observation.images.top"),
]
EVENTS_AT_2 = [
    event_row("user", "skip wiping", "interjection"),
    event_row(
        "assistant",
        None,
        None,
        [{"type": "function", "function": {"name": "say", "arguments": {"text": "Skipping wiping."}}}],
    ),
]
# Same emission tick, two cameras: triggers per-camera disambiguation in
# resolvers, mirroring how Module 3 of the annotation pipeline writes one
# (vqa, user) + (vqa, assistant) pair per camera.
EVENTS_AT_3_TWO_CAMERAS = [
    event_row("user", "how many cups (top)?", "vqa", camera="observation.images.top"),
    event_row("assistant", '{"count": 3}', "vqa", camera="observation.images.top"),
    event_row("user", "how many cups (wrist)?", "vqa", camera="observation.images.wrist"),
    event_row("assistant", '{"count": 1}', "vqa", camera="observation.images.wrist"),
]


def test_resolver_temporal_semantics():
    assert active_at(0.5, persistent=PERSISTENT, style="subtask")["content"] == "subtask 0"
    assert active_at(1.0, persistent=PERSISTENT, style="subtask")["content"] == "subtask 1"
    assert emitted_at(0.5, persistent=PERSISTENT, events=[], style="vqa", role="assistant") is None
    assert (
        emitted_at(1.0, persistent=PERSISTENT, events=EVENTS_AT_1, style="vqa", role="assistant")["content"]
        == '{"count": 2}'
    )


def test_persistent_relative_resolvers_reject_event_styles():
    with pytest.raises(ValueError, match="event-only"):
        active_at(1.0, persistent=PERSISTENT, style="vqa")
    with pytest.raises(ValueError, match="event-only"):
        nth_prev(1.0, persistent=PERSISTENT, style="interjection")


def test_nth_prev_and_next():
    assert nth_prev(1.0, persistent=PERSISTENT, style="subtask", offset=1)["content"] == "subtask 0"
    assert nth_next(0.0, persistent=PERSISTENT, style="subtask", offset=1)["content"] == "subtask 1"


def test_substitution_if_present_multimodal_and_tool_calls():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(
                role="user",
                content=[
                    {"type": "image", "feature": "observation.images.top"},
                    {"type": "text", "text": "${task}: ${interjection}"},
                ],
                stream="high_level",
                if_present="interjection",
            ),
            MessageTurn(
                role="assistant",
                content="${plan}",
                stream="high_level",
                target=True,
                tool_calls_from="speech",
            ),
        ],
        bindings={"plan": "active_at(t, style=plan)"},
    )

    rendered = render_sample(
        recipe=recipe,
        persistent=PERSISTENT,
        events=EVENTS_AT_2,
        t=2.0,
        sample_idx=0,
        task="clean kitchen",
    )

    assert rendered["messages"][0]["content"][1]["text"] == "clean kitchen: skip wiping"
    assert rendered["messages"][1]["content"] == "plan 0"
    assert rendered["messages"][1]["tool_calls"][0]["function"]["name"] == "say"
    assert rendered["message_streams"] == ["high_level", "high_level"]
    assert rendered["target_message_indices"] == [1]


def test_exact_event_miss_returns_none_when_target_skips():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${vqa_query}", stream="high_level", if_present="vqa_query"),
            MessageTurn(
                role="assistant",
                content="${vqa}",
                stream="high_level",
                target=True,
                if_present="vqa",
            ),
        ]
    )

    assert (
        render_sample(recipe=recipe, persistent=PERSISTENT, events=EVENTS_AT_2, t=0.0, sample_idx=0) is None
    )


def test_deterministic_blend_sampling():
    recipe = TrainingRecipe(
        blend={
            "a": TrainingRecipe(
                weight=1.0,
                messages=[
                    MessageTurn(role="user", content="${task}", stream="high_level"),
                    MessageTurn(role="assistant", content="a", stream="high_level", target=True),
                ],
            ),
            "b": TrainingRecipe(
                weight=1.0,
                messages=[
                    MessageTurn(role="user", content="${task}", stream="high_level"),
                    MessageTurn(role="assistant", content="b", stream="high_level", target=True),
                ],
            ),
        }
    )

    first = render_sample(
        recipe=recipe, persistent=PERSISTENT, events=EVENTS_AT_2, t=0.0, sample_idx=123, task="x"
    )
    second = render_sample(
        recipe=recipe, persistent=PERSISTENT, events=EVENTS_AT_2, t=0.0, sample_idx=123, task="x"
    )
    assert first == second


def test_emitted_at_filters_vqa_by_camera():
    top = emitted_at(
        3.0,
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        style="vqa",
        role="assistant",
        camera="observation.images.top",
    )
    wrist = emitted_at(
        3.0,
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        style="vqa",
        role="assistant",
        camera="observation.images.wrist",
    )
    assert top["content"] == '{"count": 3}'
    assert wrist["content"] == '{"count": 1}'


def test_emitted_at_raises_on_ambiguous_per_camera_vqa():
    with pytest.raises(ValueError, match="Ambiguous resolver"):
        emitted_at(
            3.0,
            persistent=PERSISTENT,
            events=EVENTS_AT_3_TWO_CAMERAS,
            style="vqa",
            role="assistant",
        )


def _vqa_subrecipe(camera: str) -> TrainingRecipe:
    return TrainingRecipe(
        weight=1.0,
        bindings={
            "vqa_query": f"emitted_at(t, style=vqa, role=user, camera={camera})",
            "vqa": f"emitted_at(t, style=vqa, role=assistant, camera={camera})",
        },
        messages=[
            MessageTurn(
                role="user",
                content=[{"type": "image", "feature": camera}, {"type": "text", "text": "${vqa_query}"}],
                stream="high_level",
                if_present="vqa_query",
            ),
            MessageTurn(
                role="assistant",
                content="${vqa}",
                stream="high_level",
                target=True,
                if_present="vqa",
            ),
        ],
    )


@pytest.mark.parametrize(
    ("camera", "expected_query", "expected_answer"),
    [
        ("observation.images.top", "how many cups (top)?", '{"count": 3}'),
        ("observation.images.wrist", "how many cups (wrist)?", '{"count": 1}'),
    ],
)
def test_per_camera_blend_renders_both_views(camera, expected_query, expected_answer):
    rendered = render_sample(
        recipe=_vqa_subrecipe(camera),
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        t=3.0,
        sample_idx=0,
    )

    assert rendered["messages"][0]["content"][0]["feature"] == camera
    assert rendered["messages"][0]["content"][1]["text"] == expected_query
    assert rendered["messages"][1]["content"] == expected_answer


def test_resolve_task_picks_rephrasing_deterministically_per_sample():
    rephrasings = [
        persistent_row("user", "tidy the kitchen", "task_aug", 0.0),
        persistent_row("user", "please clean up the kitchen", "task_aug", 0.0),
        persistent_row("user", "kitchen needs tidying", "task_aug", 0.0),
        persistent_row("user", "make the kitchen clean", "task_aug", 0.0),
    ]
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="ok", stream="high_level", target=True),
        ]
    )

    # No explicit task override → resolver consults persistent rows.
    seen: set[str] = set()
    for sample_idx in range(64):
        rendered = render_sample(
            recipe=recipe,
            persistent=rephrasings,
            events=[],
            t=0.0,
            sample_idx=sample_idx,
            dataset_ctx={"task": "canonical kitchen task"},
        )
        seen.add(rendered["messages"][0]["content"])
    # Every rephrasing should be reachable across enough samples.
    assert seen == {r["content"] for r in rephrasings}
    # Same sample_idx → same pick (determinism).
    a = render_sample(
        recipe=recipe,
        persistent=rephrasings,
        events=[],
        t=0.0,
        sample_idx=42,
        dataset_ctx={"task": "canonical"},
    )
    b = render_sample(
        recipe=recipe,
        persistent=rephrasings,
        events=[],
        t=0.0,
        sample_idx=42,
        dataset_ctx={"task": "canonical"},
    )
    assert a["messages"][0]["content"] == b["messages"][0]["content"]


def test_resolve_task_falls_back_to_canonical_without_rephrasings():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="ok", stream="high_level", target=True),
        ]
    )
    rendered = render_sample(
        recipe=recipe,
        persistent=PERSISTENT,  # no task_aug rows
        events=[],
        t=0.0,
        sample_idx=0,
        dataset_ctx={"task": "clean the kitchen"},
    )
    assert rendered["messages"][0]["content"] == "clean the kitchen"


def test_resolve_task_explicit_override_beats_rephrasings():
    rephrasings = [
        persistent_row("user", "rephrased one", "task_aug", 0.0),
        persistent_row("user", "rephrased two", "task_aug", 0.0),
    ]
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="ok", stream="high_level", target=True),
        ]
    )
    rendered = render_sample(
        recipe=recipe,
        persistent=rephrasings,
        events=[],
        t=0.0,
        sample_idx=0,
        task="explicit override wins",
        dataset_ctx={"task": "canonical"},
    )
    assert rendered["messages"][0]["content"] == "explicit override wins"


def test_emitted_at_persistent_tolerates_small_timestamp_drift():
    """Persistent ``emitted_at`` should match within EMITTED_AT_TOLERANCE_S
    so callers that derive ``t`` arithmetically (``frame_idx / fps``) still
    line up with the parquet-stored timestamp.
    """
    rows = [persistent_row("assistant", "memo", "memory", 1.0)]
    # Half a tolerance window — bit-different float, comfortably inside
    inside = emitted_at(1.0 + EMITTED_AT_TOLERANCE_S / 2, persistent=rows, events=[], style="memory")
    assert inside is not None and inside["content"] == "memo"

    # Just past the window — no match
    outside = emitted_at(1.0 + EMITTED_AT_TOLERANCE_S * 2, persistent=rows, events=[], style="memory")
    assert outside is None


def test_render_sample_rejects_non_dict_language_rows():
    """``_normalize_rows`` must surface malformed inputs as TypeError.

    A pipeline that hands the renderer a non-dict (e.g. a stray string)
    is a real upstream bug — silent skipping would let it propagate.
    """
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="ok", stream="high_level", target=True),
        ]
    )
    with pytest.raises(TypeError, match="must be dictionaries"):
        render_sample(
            recipe=recipe,
            persistent=["not a dict"],
            events=[],
            t=0.0,
            sample_idx=0,
            task="x",
        )


def test_low_level_branch_renders_active_subtask():
    low_level = TrainingRecipe(
        blend={
            "low": TrainingRecipe(
                weight=1.0,
                messages=[
                    MessageTurn(
                        role="user",
                        content="${task}\nPlan: ${plan}\nMemory: ${memory}",
                        stream="high_level",
                    ),
                    MessageTurn(
                        role="assistant",
                        content="${subtask}",
                        stream="low_level",
                        target=True,
                    ),
                ],
            )
        }
    )

    rendered = render_sample(
        recipe=low_level,
        persistent=PERSISTENT,
        events=[],
        t=0.5,
        sample_idx=0,
        task="clean kitchen",
    )

    assert rendered["messages"][-1] == {"role": "assistant", "content": "subtask 0"}
    assert rendered["message_streams"][-1] == "low_level"
    assert rendered["target_message_indices"] == [1]
