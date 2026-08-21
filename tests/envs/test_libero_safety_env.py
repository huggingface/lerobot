import pytest
import torch

from lerobot.envs.factory import make_env_config
from lerobot.envs.libero import SAFETY_SUITES, LiberoEnv, _get_suite, get_task_init_states


class _FakeTask:
    def __init__(self, name, language, problem_folder, bddl_file, init_states_file, level):
        self.name = name
        self.language = language
        self.problem_folder = problem_folder
        self.bddl_file = bddl_file
        self.init_states_file = init_states_file
        self.level = level


class _FakeSuite:
    def __init__(self, tasks):
        self.tasks = tasks

    def get_task(self, i):
        return self.tasks[i]


def test_get_task_init_states_inserts_level_dir_for_libero_safety(monkeypatch, tmp_path):
    """LIBERO-Safety nests init states one level deeper than stock LIBERO:
    init_files/<suite>/L{level}/<task>.pruned_init. Uses a real file under tmp_path so the
    new path-existence validation (see `_require_libero_file`) also gets exercised, not just
    the string construction."""
    task = _FakeTask(
        name="t0",
        language="do thing",
        problem_folder="human_safety",
        bddl_file="do_thing.bddl",
        init_states_file="do_thing.pruned_init",
        level=2,
    )
    suite = _FakeSuite([task])

    init_dir = tmp_path / "init_states" / "human_safety" / "L2"
    init_dir.mkdir(parents=True)
    real_path = init_dir / "do_thing.pruned_init"
    torch.save(torch.zeros(1, 3), real_path)

    monkeypatch.setattr(
        "lerobot.envs.libero.get_libero_path",
        lambda kind: str(tmp_path / "init_states") if kind == "init_states" else "/fake/other",
    )

    result = get_task_init_states(suite, 0, is_libero_safety=True)

    torch.testing.assert_close(result, torch.zeros(1, 3))


def test_get_task_init_states_libero_safety_missing_file_raises_with_actionable_message(
    monkeypatch, tmp_path
):
    task = _FakeTask(
        name="t0",
        language="do thing",
        problem_folder="human_safety",
        bddl_file="do_thing.bddl",
        init_states_file="do_thing.pruned_init",
        level=2,
    )
    suite = _FakeSuite([task])

    missing_root = tmp_path / "init_states"
    monkeypatch.setattr(
        "lerobot.envs.libero.get_libero_path",
        lambda kind: str(missing_root) if kind == "init_states" else "/fake/other",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        get_task_init_states(suite, 0, is_libero_safety=True)

    message = str(exc_info.value)
    expected_path = missing_root / "human_safety" / "L2" / "do_thing.pruned_init"
    assert str(expected_path) in message
    assert "config.yaml" in message
    assert "init_states" in message


def test_get_task_init_states_stock_path_unaffected(monkeypatch, tmp_path):
    """Stock LIBERO suites (is_libero_safety=False) must keep the flat, un-leveled path."""
    task = _FakeTask(
        name="t0",
        language="do thing",
        problem_folder="libero_10",
        bddl_file="do_thing.bddl",
        init_states_file="do_thing.pruned_init",
        level=0,
    )
    suite = _FakeSuite([task])

    monkeypatch.setattr(
        "lerobot.envs.libero.get_libero_path",
        lambda kind: "/fake/init_states" if kind == "init_states" else "/fake/other",
    )
    captured = {}

    def fake_torch_load(path, weights_only=False):
        captured["path"] = path
        return "loaded"

    monkeypatch.setattr(torch, "load", fake_torch_load)

    # Stock (non-safety) path never goes through _require_libero_file, so no real file needed.
    get_task_init_states(suite, 0)

    from pathlib import Path

    assert captured["path"] == Path("/fake/init_states/libero_10/do_thing.pruned_init")


def test_libero_env_bddl_path_inserts_level_dir_for_libero_safety(tmp_path, monkeypatch):
    task = _FakeTask(
        name="t0",
        language="do thing",
        problem_folder="obstacle_avoidance",
        bddl_file="do_thing.bddl",
        init_states_file="do_thing.pruned_init",
        level=1,
    )
    suite = _FakeSuite([task])

    bddl_dir = tmp_path / "bddl_files" / "obstacle_avoidance" / "L1"
    bddl_dir.mkdir(parents=True)
    (bddl_dir / "do_thing.bddl").write_text("(define (problem test))")

    monkeypatch.setattr(
        "lerobot.envs.libero.get_libero_path",
        lambda kind: str(tmp_path / "bddl_files") if kind == "bddl_files" else "/fake/other",
    )

    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="obstacle_avoidance",
        init_states=False,
        is_libero_safety=True,
    )

    assert env._task_bddl_file == str(bddl_dir / "do_thing.bddl")


def test_libero_env_bddl_path_missing_file_raises_with_actionable_message(tmp_path, monkeypatch):
    task = _FakeTask(
        name="t0",
        language="do thing",
        problem_folder="obstacle_avoidance",
        bddl_file="do_thing.bddl",
        init_states_file="do_thing.pruned_init",
        level=1,
    )
    suite = _FakeSuite([task])

    monkeypatch.setattr(
        "lerobot.envs.libero.get_libero_path",
        lambda kind: str(tmp_path / "bddl_files") if kind == "bddl_files" else "/fake/other",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        LiberoEnv(
            task_suite=suite,
            task_id=0,
            task_suite_name="obstacle_avoidance",
            init_states=False,
            is_libero_safety=True,
        )

    message = str(exc_info.value)
    expected_path = tmp_path / "bddl_files" / "obstacle_avoidance" / "L1" / "do_thing.bddl"
    assert str(expected_path) in message
    assert "Dockerfile.benchmark.libero_safety" in message


def test_libero_env_bddl_path_unaffected_for_stock_libero(monkeypatch):
    task = _FakeTask(
        name="t0",
        language="do thing",
        problem_folder="libero_spatial",
        bddl_file="do_thing.bddl",
        init_states_file="do_thing.pruned_init",
        level=0,
    )
    suite = _FakeSuite([task])

    monkeypatch.setattr(
        "lerobot.envs.libero.get_libero_path",
        lambda kind: "/fake/bddl_files" if kind == "bddl_files" else "/fake/other",
    )

    # Stock (non-safety) path never goes through _require_libero_file, so no real file needed.
    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_spatial",
        init_states=False,
    )

    assert env._task_bddl_file == "/fake/bddl_files/libero_spatial/do_thing.bddl"


def test_libero_safety_env_config_registered_with_expected_defaults():
    cfg = make_env_config("libero_safety")

    assert cfg.is_libero_safety is True
    assert cfg.task == "affordance,human_safety,obstacle_avoidance"
    assert cfg.level is None
    assert cfg.task_ids is None
    assert cfg.gym_kwargs["control_freq"] == cfg.fps


@pytest.mark.parametrize(
    "level,expected_task_ids",
    [
        (0, [0, 1, 2, 3, 4]),
        (1, [5, 6, 7, 8, 9]),
        (2, [10, 11, 12, 13, 14]),
        ([0, 2], [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]),
    ],
)
def test_libero_safety_env_level_computes_task_ids(level, expected_task_ids):
    cfg = make_env_config("libero_safety", task="affordance", level=level)
    assert cfg.task_ids == expected_task_ids


def test_libero_safety_env_invalid_level_raises():
    with pytest.raises(ValueError, match=r"Invalid LIBERO-Safety level: 4.*0, 1, 2"):
        make_env_config("libero_safety", level=4)


def test_libero_safety_env_invalid_suite_raises():
    with pytest.raises(ValueError, match=r"Invalid LIBERO-Safety suite\(s\).*not_a_suite"):
        make_env_config("libero_safety", task="not_a_suite")


def test_libero_safety_env_level_and_task_ids_conflict_raises():
    with pytest.raises(ValueError, match=r"either --env.level or --env.task_ids"):
        make_env_config("libero_safety", level=1, task_ids=[0])


def test_libero_safety_env_accepts_all_five_suites():
    suites = (
        "affordance",
        "human_safety",
        "obstacle_avoidance",
        "obstacle_avoidance_human",
        "reasoning_safety",
    )
    for suite in suites:
        cfg = make_env_config("libero_safety", task=suite)
        assert cfg.task == suite


def test_libero_safety_env_assets_repo_id_default_and_override():
    assert make_env_config("libero_safety").assets_repo_id == "LIBERO-Safety/libero_safety_assets"
    cfg = make_env_config("libero_safety", assets_repo_id="some-org/some-mirror")
    assert cfg.assets_repo_id == "some-org/some-mirror"


def test_get_suite_raises_clear_error_when_stock_libero_installed_instead_of_fork():
    """Stock hf-libero and the LIBERO-Safety fork both import as `libero`, so a suite name
    from SAFETY_SUITES missing out of the real, installed benchmark.get_benchmark_dict()
    (as is the case for stock hf-libero, exercised here for real, not mocked) must raise a
    message that points at "wrong package installed", not the generic "unknown suite"."""
    with pytest.raises(RuntimeError, match="standard LIBERO was imported"):
        _get_suite("affordance", is_libero_safety=True)


def test_get_suite_is_libero_safety_false_still_raises_plain_unknown_suite_error():
    with pytest.raises(ValueError, match="Unknown LIBERO suite 'affordance'"):
        _get_suite("affordance", is_libero_safety=False)


def test_get_suite_stock_suites_unaffected_by_is_libero_safety_flag():
    assert _get_suite("libero_spatial", is_libero_safety=False).tasks
    # SAFETY_SUITES sanity-checked against the module the tests already import.
    assert {
        "affordance",
        "human_safety",
        "obstacle_avoidance",
        "obstacle_avoidance_human",
        "reasoning_safety",
    } == SAFETY_SUITES
