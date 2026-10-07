"""Language-input contract of the VLABench env wrapper: the policy must receive the simulator's
referent-bearing natural-language instruction in the dataset's task-string format, never a silent
task-name fallback."""

import logging
import random

from lerobot.envs.vlabench import resolve_task_language


def _task(module: str, instruction, **attrs):
    cls = type(
        "StubTask",
        (),
        {
            "get_instruction": lambda self: (
                random.sample([instruction], 1)[0] if instruction is not None else None
            )
        },
    )
    cls.__module__ = module
    obj = cls()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_native_instruction_gets_dataset_primitive_prefix():
    t = _task(
        "VLABench.tasks.hierarchical_tasks.primitive.select_poker_series", "Please pick the poker 7 of spades"
    )
    assert resolve_task_language(t, "select_poker") == (
        "primitive: Please pick the poker 7 of spades",
        "native_get_instruction",
    )


def test_composite_family_prefix():
    t = _task("VLABench.tasks.hierarchical_tasks.composite.cluster", "Cluster the objects into two classes.")
    assert resolve_task_language(t, "cluster")[0] == "composite: Cluster the objects into two classes."


def test_prefix_never_inserted_twice():
    t = _task("VLABench.tasks.hierarchical_tasks.primitive.x", "primitive: Take me a bottle of cola")
    assert resolve_task_language(t, "x")[0] == "primitive: Take me a bottle of cola"


def test_provider_attribute_used_verbatim():
    t = _task("VLABench.tasks.x", "ignored", task_description="Do the thing")
    assert resolve_task_language(t, "x") == ("Do the thing", "task_description")


def test_native_lookup_preserves_global_rng_stream():
    t = _task("VLABench.tasks.hierarchical_tasks.primitive.x", "Please pick the poker 2 of hearts")
    random.seed(123)
    resolve_task_language(t, "x")
    after = random.random()
    random.seed(123)
    assert random.random() == after


def test_task_name_fallback_is_loud(caplog):
    t = _task("VLABench.tasks.x", None)
    with caplog.at_level(logging.WARNING):
        text, source = resolve_task_language(t, "select_book")
    assert (text, source) == ("select_book", "task_name_fallback")
    assert "FALLING BACK TO THE TASK NAME" in caplog.text
