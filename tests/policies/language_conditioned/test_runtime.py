from lerobot.policies.language_conditioned import (
    LanguageConditionedRuntime,
    RuntimeState,
    ToolCall,
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
        return "new plan <say>ok</say>"

    def parse_tool_calls(self, text):
        assert text == "new plan <say>ok</say>"
        return [ToolCall("say", {"text": "ok"})]

    def answer_vqa(self, question, camera, observation, state):
        return VQAResult(answer=f"answer: {question}")

    def update_language_state(self, observation, state):
        self.updated = True
        state.set_context("subtask", "pick cup", label="subtask")


class FakeTool:
    def __init__(self):
        self.calls = []

    def call(self, args):
        self.calls.append(args)


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


def test_runtime_handles_user_interjection_and_dispatches_tools():
    adapter = FakeAdapter()
    tool = FakeTool()
    runtime = LanguageConditionedRuntime(
        policy_adapter=adapter,
        observation_provider=lambda: {"observation.state": 1},
        tools={"say": tool},
    )
    runtime.set_task("clean")
    runtime.state.extra["recent_interjection"] = "please say ok"
    runtime.state.emit("user_interjection")

    logs = runtime.step_once()

    assert ("interjection", "please say ok") in adapter.text_calls
    assert runtime.state.language_context["plan"] == "new plan <say>ok</say>"
    assert tool.calls == [{"text": "ok"}]
    assert "  speech: ok" in logs


def test_runtime_state_aliases_legacy_keys_to_language_context():
    state = RuntimeState()
    state["current_subtask"] = "open drawer"
    state["current_memory"] = "drawer open"

    assert state.get("current_subtask") == "open drawer"
    assert state.language_context == {"subtask": "open drawer", "memory": "drawer open"}
