from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import BaseLanguageAdapter, GenerationConfig, looks_like_gibberish


class ScriptedAdapter(BaseLanguageAdapter):
    """Base adapter whose text generation returns queued strings per kind."""

    def __init__(self, scripts, gen=None):
        super().__init__(policy=object(), gen=gen)
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls = []

    def select_action(self, observation, state):
        return None

    def generate_text(self, kind, observation, state, user_text=None):
        self.calls.append(kind)
        queue = self.scripts.get(kind, [])
        return queue.pop(0) if queue else ""


def test_cascade_sets_subtask_then_memory():
    adapter = ScriptedAdapter({"subtask": ["pick the red cup"], "memory": ["the cup is grasped"]})
    state = RuntimeState(task="clean")

    adapter.update_language_state(None, state)

    assert state.language_context["subtask"] == "pick the red cup"
    assert state.language_context["memory"] == "the cup is grasped"
    assert adapter.calls == ["subtask", "memory"]


def test_gibberish_subtask_is_rejected_and_counted():
    adapter = ScriptedAdapter({"subtask": [":::: ::"], "memory": ["should not run"]})
    state = RuntimeState(task="clean")

    adapter.update_language_state(None, state)

    assert "subtask" not in state.language_context
    assert adapter.diag.gibberish.get("subtask") == 1
    assert adapter.calls == ["subtask"]  # memory never generated when subtask is rejected


def test_throttle_regenerates_every_n_chunks():
    adapter = ScriptedAdapter(
        {
            "subtask": ["pick the first cup", "pick the second cup"],
            "memory": ["memory one two three", "memory four five six"],
        },
        gen=GenerationConfig(chunks_per_regen=2),
    )
    state = RuntimeState(task="clean")

    adapter.update_language_state(None, state)  # generates
    assert state.language_context["subtask"] == "pick the first cup"
    adapter.update_language_state(None, state)  # throttled — no generation
    assert state.language_context["subtask"] == "pick the first cup"
    adapter.update_language_state(None, state)  # generates again
    assert state.language_context["subtask"] == "pick the second cup"


def test_handle_interjection_sets_plan_and_strips_say():
    adapter = ScriptedAdapter({"interjection": ["turn to the left now <say>heading left</say>"]})
    state = RuntimeState(task="clean")

    adapter.handle_interjection("turn", None, state)

    assert state.language_context["plan"] == "turn to the left now"


def test_looks_like_gibberish_basic():
    assert looks_like_gibberish("")
    assert looks_like_gibberish(":::: ::")
    assert not looks_like_gibberish("pick up the red cube")
