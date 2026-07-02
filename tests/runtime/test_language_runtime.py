from lerobot.runtime import (
    LanguageConditionedRuntime,
    RuntimeState,
    VQAResult,
)


class FakeAdapter:
    def __init__(self):
        self.updated = False
        self.text_calls = []

    def select_action(self, observation, state):
        assert observation == {"observation.state": 1}
        assert state.task == "clean"
        return ["a0", "a1"]

    def select_text(self, kind, observation, state, user_text=None):
        self.text_calls.append((kind, user_text))
        return "new plan"

    def answer_vqa(self, question, camera, observation, state):
        return VQAResult(answer=f"answer: {question}")

    def update_language_state(self, observation, state):
        self.updated = True
        state.set_context("subtask", "pick cup", label="subtask")


def test_runtime_tick_updates_language_enqueues_and_dispatches_action():
    adapter = FakeAdapter()
    executed = []
    runtime = LanguageConditionedRuntime(
        policy_adapter=adapter,
        observation_provider=lambda: {"observation.state": 1},
        action_executor=executed.append,
    )
    runtime.set_task("clean")

    logs = runtime.step_once()

    assert adapter.updated
    assert runtime.state.language_context["subtask"] == "pick cup"
    assert executed == ["a0"]
    assert list(runtime.state.action_queue) == ["a1"]
    assert "  subtask: pick cup" in logs


def test_runtime_handles_user_interjection():
    adapter = FakeAdapter()
    runtime = LanguageConditionedRuntime(
        policy_adapter=adapter,
        observation_provider=lambda: {"observation.state": 1},
    )
    runtime.set_task("clean")
    runtime.state.extra["recent_interjection"] = "please say ok"
    runtime.state.emit("user_interjection")

    runtime.step_once()

    assert ("interjection", "please say ok") in adapter.text_calls
    assert runtime.state.language_context["plan"] == "new plan"


def test_runtime_state_aliases_legacy_keys_to_language_context():
    state = RuntimeState()
    state["current_subtask"] = "open drawer"
    state["current_memory"] = "drawer open"

    assert state.get("current_subtask") == "open drawer"
    assert state.language_context == {"subtask": "open drawer", "memory": "drawer open"}
