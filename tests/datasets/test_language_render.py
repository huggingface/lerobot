#!/usr/bin/env python

from pathlib import Path

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.datasets.language_render import active_at, emitted_at, nth_next, nth_prev, render_sample


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


def test_per_camera_blend_renders_both_views():
    recipe = TrainingRecipe(
        blend={
            "top": TrainingRecipe(
                weight=1.0,
                bindings={
                    "vqa_query": ("emitted_at(t, style=vqa, role=user, camera=observation.images.top)"),
                    "vqa": ("emitted_at(t, style=vqa, role=assistant, camera=observation.images.top)"),
                },
                messages=[
                    MessageTurn(
                        role="user",
                        content=[
                            {"type": "image", "feature": "observation.images.top"},
                            {"type": "text", "text": "${vqa_query}"},
                        ],
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
            ),
            "wrist": TrainingRecipe(
                weight=1.0,
                bindings={
                    "vqa_query": ("emitted_at(t, style=vqa, role=user, camera=observation.images.wrist)"),
                    "vqa": ("emitted_at(t, style=vqa, role=assistant, camera=observation.images.wrist)"),
                },
                messages=[
                    MessageTurn(
                        role="user",
                        content=[
                            {"type": "image", "feature": "observation.images.wrist"},
                            {"type": "text", "text": "${vqa_query}"},
                        ],
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
            ),
        }
    )

    rendered_top = render_sample(
        recipe=recipe.blend["top"],
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        t=3.0,
        sample_idx=0,
    )
    rendered_wrist = render_sample(
        recipe=recipe.blend["wrist"],
        persistent=PERSISTENT,
        events=EVENTS_AT_3_TWO_CAMERAS,
        t=3.0,
        sample_idx=0,
    )

    assert rendered_top["messages"][0]["content"][0]["feature"] == "observation.images.top"
    assert rendered_top["messages"][0]["content"][1]["text"] == "how many cups (top)?"
    assert rendered_top["messages"][1]["content"] == '{"count": 3}'

    assert rendered_wrist["messages"][0]["content"][0]["feature"] == "observation.images.wrist"
    assert rendered_wrist["messages"][0]["content"][1]["text"] == "how many cups (wrist)?"
    assert rendered_wrist["messages"][1]["content"] == '{"count": 1}'


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


def test_flow_only_low_level_recipe_renders_without_target():
    """Regression: a flow-only ``low_level`` recipe has no ``target`` turn —
    its supervision is the action-expert flow loss, not text-CE. It must
    still render (not ``None``), otherwise every blend draw of it is dropped
    and the action expert never receives a flow loss."""
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(
                role="user",
                content="${subtask}",
                stream="low_level",
                if_present="subtask",
            ),
        ],
        bindings={"subtask": "active_at(t, style=subtask)"},
    )

    rendered = render_sample(
        recipe=recipe,
        persistent=PERSISTENT,
        events=[],
        t=0.5,
        sample_idx=0,
        task="clean kitchen",
    )

    assert rendered is not None
    assert rendered["messages"] == [{"role": "user", "content": "subtask 0"}]
    assert rendered["message_streams"] == ["low_level"]
    assert rendered["target_message_indices"] == []


def test_vqa_frame_is_consumed_over_the_weighted_blend():
    """A frame carrying a VQA annotation renders the ``ask_vqa*`` sub-recipe
    even when its blend weight is tiny — VQA annotations are sparse and must
    never be wasted on a subtask/action draw."""
    recipe = TrainingRecipe(
        blend={
            "high_level_subtask": TrainingRecipe(
                weight=0.99,
                messages=[
                    MessageTurn(role="user", content="${task}", stream="high_level"),
                    MessageTurn(role="assistant", content="a subtask", stream="high_level", target=True),
                ],
            ),
            "ask_vqa_top": TrainingRecipe(
                weight=0.01,
                bindings={
                    "vqa_query": "emitted_at(t, style=vqa, role=user, camera=observation.images.top)",
                    "vqa": "emitted_at(t, style=vqa, role=assistant, camera=observation.images.top)",
                },
                messages=[
                    MessageTurn(
                        role="user", content="${vqa_query}", stream="high_level", if_present="vqa_query"
                    ),
                    MessageTurn(
                        role="assistant",
                        content="${vqa}",
                        stream="high_level",
                        target=True,
                        if_present="vqa",
                    ),
                ],
            ),
        }
    )
    # A frame WITH a vqa event renders VQA on every sample_idx, despite the
    # ask_vqa weight being only 0.01.
    for sample_idx in range(20):
        rendered = render_sample(
            recipe=recipe, persistent=PERSISTENT, events=EVENTS_AT_1, t=1.0, sample_idx=sample_idx, task="x"
        )
        assert rendered["messages"][-1]["content"] == '{"count": 2}', sample_idx
    # A frame WITHOUT a vqa event falls back to the normal weighted blend.
    rendered = render_sample(recipe=recipe, persistent=PERSISTENT, events=[], t=1.0, sample_idx=0, task="x")
    assert rendered["messages"][-1]["content"] == "a subtask"


def test_canonical_recipe_can_render_low_level_branch():
    recipe = TrainingRecipe.from_yaml(Path("src/lerobot/configs/recipes/pi05_hirobot.yaml"))
    low_level = TrainingRecipe(blend={"low": recipe.blend["low_level_execution"]})

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
