from types import SimpleNamespace

from lerobot.policies.pi052.inference.pi052_adapter import PI052PolicyAdapter, split_plan_and_say
from lerobot.runtime import RuntimeState


def test_pi052_adapter_builds_recipe_prompts_from_runtime_state():
    adapter = PI052PolicyAdapter(policy=object())
    state = RuntimeState(
        task="clean the kitchen",
        language_context={"memory": "cup moved", "plan": "pick then place"},
        extra={"prior_subtask": "pick the cup"},
    )

    assert adapter.messages_for("subtask", state) == [{"role": "user", "content": "clean the kitchen"}]
    assert adapter.messages_for("memory", state) == [
        {"role": "user", "content": "clean the kitchen"},
        {"role": "assistant", "content": "Previous memory: cup moved"},
        {"role": "user", "content": "Completed subtask: pick the cup"},
    ]
    assert adapter.messages_for("interjection", state, user_text="wait") == [
        {"role": "user", "content": "clean the kitchen"},
        {"role": "assistant", "content": "Previous plan:\npick then place"},
        {"role": "user", "content": "wait"},
    ]
    assert adapter.messages_for("vqa", state, user_text="where is the cup?") == [
        {"role": "user", "content": "where is the cup?"}
    ]


def test_pi052_adapter_strips_say_markers_from_plan_text():
    adapter = PI052PolicyAdapter(policy=object())
    text = "Move to the sink. <say>heading to the sink</say>"

    assert split_plan_and_say(text) == ("Move to the sink.", "heading to the sink")
    assert adapter.plan_from_text(text) == "Move to the sink."


def test_pi052_runtime_cli_smoke_does_not_load_model(monkeypatch):
    from lerobot.policies.pi052.inference import runtime_cli

    fake_policy = SimpleNamespace(config=SimpleNamespace(device="cpu"))

    monkeypatch.setattr(
        runtime_cli,
        "_load_policy_and_preprocessor",
        lambda policy_path, dataset_repo_id: (fake_policy, None, None, None),
    )
    monkeypatch.setattr(runtime_cli, "_run_repl", lambda runtime, initial_task, max_ticks: 0)

    assert runtime_cli.main(["--policy.path=fake", "--no_robot", "--task=clean", "--max_ticks=0"]) == 0
