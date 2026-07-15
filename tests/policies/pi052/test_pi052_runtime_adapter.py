from types import SimpleNamespace

from lerobot.policies.pi052.inference.pi052_adapter import PI052PolicyAdapter
from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import split_plan_and_say


def test_pi052_adapter_builds_recipe_prompts_from_runtime_state():
    adapter = PI052PolicyAdapter(policy=object())
    state = RuntimeState(
        task="clean the kitchen",
        language_context={"memory": "cup moved", "plan": "pick then place"},
        extra={"prior_subtask": "pick the cup"},
    )

    assert adapter.build_messages("subtask", state) == [{"role": "user", "content": "clean the kitchen"}]
    assert adapter.build_messages("memory", state) == [
        {"role": "user", "content": "clean the kitchen"},
        {"role": "assistant", "content": "Previous memory: cup moved"},
        {"role": "user", "content": "Completed subtask: pick the cup"},
    ]
    assert adapter.build_messages("interjection", state, user_text="wait") == [
        {"role": "user", "content": "clean the kitchen"},
        {"role": "assistant", "content": "Previous plan:\npick then place"},
        {"role": "user", "content": "wait"},
    ]


def test_pi052_adapter_strips_say_markers_from_plan_text():
    adapter = PI052PolicyAdapter(policy=object())
    text = "Move to the sink. <say>heading to the sink</say>"

    assert split_plan_and_say(text) == ("Move to the sink.", "heading to the sink")
    assert adapter.plan_from_text(text) == "Move to the sink."


def test_rollout_language_cli_smoke_does_not_load_model(monkeypatch):
    """lerobot-rollout dispatches language flags to the adapter-based runtime."""
    from lerobot.runtime import cli
    from lerobot.scripts import lerobot_rollout

    fake_policy = SimpleNamespace(config=SimpleNamespace(device="cpu", type="pi052"))

    monkeypatch.setattr(
        cli,
        "_load_policy_and_preprocessor",
        lambda policy_path, **kwargs: (fake_policy, None, None),
    )
    monkeypatch.setattr(cli, "_run_repl", lambda runtime, **kwargs: 0)

    assert lerobot_rollout.main(["--policy.path=fake", "--no_robot", "--task=clean", "--max_ticks=0"]) == 0


def test_rollout_language_dispatch_preserves_standard_molmoact2_path(monkeypatch):
    """MolmoAct2 only opts into open prompting when a language flag is present."""
    from lerobot.scripts import lerobot_rollout

    standard = [
        "--policy.path=lerobot/MolmoAct2-SO100_101-LeRobot",
        "--robot.type=so101_follower",
        "--task=pick up the cube",
    ]
    assert not lerobot_rollout._uses_language_runtime(standard)
    assert lerobot_rollout._uses_language_runtime([*standard, "--direct_subtask"])
    assert lerobot_rollout._uses_language_runtime(["--policy.path=lerobot/pi052_robocasa", "--sim"])

    standard_calls = []
    monkeypatch.setattr(lerobot_rollout, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(lerobot_rollout, "rollout", lambda: standard_calls.append(True))
    lerobot_rollout.main(standard)
    assert standard_calls == [True]
