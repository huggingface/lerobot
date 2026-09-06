import numpy as np
import pytest
import torch

from lerobot.action_semantics.registry import (
    get_policy_action_contract,
    get_env_action_contract,
    resolve_dataset_contract_from_repo_id,
)
from lerobot.action_semantics.adapter import ActionAdapter
from lerobot.action_semantics.conversion import to_canonical_action
from lerobot.action_semantics.conversion import convert_action
from lerobot.action_semantics.contracts import ActionContract
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.constants import OBS_ENV_STATE


def _import_diffusion_or_skip():
    try:
        from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy

        return DiffusionConfig, DiffusionPolicy
    except Exception:
        pytest.skip("diffusers or diffusion dependencies not available; skip integration test")


def make_act_policy():
    # Minimal ACT config: provide an env_state input and a 7-dim action output
    cfg = ACTConfig(
        input_features={
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(1,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    return ACTPolicy(cfg)


def test_act_policy_adapter_roundtrip_and_shape():
    policy = make_act_policy()
    batch = {OBS_ENV_STATE: torch.zeros(1, 1), "observation.state": torch.zeros(1, 1)}
    action = policy.select_action(batch)
    assert action.shape[-1] == 7

    # Resolve policy contract and run through the ActionAdapter
    policy_contract = get_policy_action_contract(policy)
    adapter = ActionAdapter()
    canonical = adapter.policy_to_canonical(action, policy_contract=policy_contract)
    env_contract = get_env_action_contract("libero")
    env_action = adapter.canonical_to_environment(canonical, env_contract=env_contract)

    # Converting env_action back to canonical should recover original canonical (no double conversion)
    back = to_canonical_action(env_action, env_contract)
    torch.testing.assert_close(back, canonical)


def test_recording_contract_encoding():
    # canonical translation 0.02 m in first translation component
    canonical = torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    adapter = ActionAdapter()
    recorded_libero = adapter.canonical_to_dataset(canonical, dataset_repo_id="libero")
    recorded_safety = adapter.canonical_to_dataset(canonical, dataset_repo_id="libero_safety")

    # LIBERO dataset stores controller command: 0.02 / 0.05 = 0.4
    assert np.isclose(recorded_libero[..., 0].item(), 0.4)
    # LIBERO-Safety stores physical delta unchanged
    assert np.isclose(recorded_safety[..., 0].item(), 0.02)


def test_no_double_conversion_safety():
    # canonical 0.02 m -> safety env controller command should be 0.01
    canonical = torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    adapter = ActionAdapter()
    env_action = adapter.canonical_to_environment(canonical, env_contract=get_env_action_contract("libero_safety"))
    # env_action first translation element should be 0.01
    assert np.isclose(env_action[..., 0].item(), 0.01)

    # converting that env action back to canonical should yield 0.02, not 0.005
    back = to_canonical_action(env_action, get_env_action_contract("libero_safety"))
    assert np.isclose(back[..., 0].item(), 0.02)


def test_policy_env_independence_preserves_physical_target():
    # canonical target
    canonical = torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    adapter = ActionAdapter()
    recorded_libero = adapter.canonical_to_dataset(canonical, dataset_repo_id="libero")
    env_safety = adapter.canonical_to_environment(canonical, env_contract=get_env_action_contract("libero_safety"))

    # values
    v_libero = recorded_libero[..., 0].item()
    e_safety = env_safety[..., 0].item()

    # check numeric relations: recorded libero 0.4, safety recorded 0.02
    assert np.isclose(v_libero * 0.05, e_safety * 2.0)
    assert np.isclose(v_libero * 0.05, 0.02)


def test_diffusion_policy_adapter_roundtrip_and_shape():
    diff_cfg_class, diff_policy_class = _import_diffusion_or_skip()
    # Minimal diffusion config with env_state and 7-dim action
    cfg = diff_cfg_class(
        input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(1,)), "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(1,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    try:
        policy = diff_policy_class(cfg)
    except ImportError:
        pytest.skip("diffusers not installed; skipping diffusion policy integration test")
    batch = {OBS_ENV_STATE: torch.zeros(1, 1), "observation.state": torch.zeros(1, 1)}
    action = policy.select_action(batch)
    assert action.shape[-1] == 7
    adapter = ActionAdapter()
    policy_contract = get_policy_action_contract(policy)
    canonical = adapter.policy_to_canonical(action, policy_contract=policy_contract)
    env_action = adapter.canonical_to_environment(canonical, env_contract=get_env_action_contract("libero"))
    back = to_canonical_action(env_action, get_env_action_contract("libero"))
    torch.testing.assert_close(back, canonical)


def test_velocity_per_step_roundtrip_and_errors():
    # conversion helpers are imported at module level

    # create a dataset contract with fps=20
    ds = resolve_dataset_contract_from_repo_id("libero_safety")
    # delta -> physical_velocity target
    delta = torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vel_contract = ActionContract(name="libero_safety_velocity", action_dim=7, fps=20.0, representation="physical_velocity")
    vel = convert_action(delta, ds, vel_contract, semantics="velocity")
    # physical delta 0.02 at 20Hz -> velocity = 0.02 * 20 = 0.4
    assert np.isclose(vel[..., 0].item(), 0.4)

    # velocity -> delta (back to dataset contract)
    back = convert_action(vel, vel_contract, ds, semantics="per_step")
    assert np.isclose(back[..., 0].item(), 0.02)

    # requesting velocity when fps is None should raise
    no_fps = ActionContract(name="no_fps", action_dim=7, fps=None, representation="physical_velocity")
    with pytest.raises(ValueError):
        convert_action(delta, no_fps, no_fps, semantics="velocity")


def test_numpy_torch_preservation():
    arr = np.ones(7, dtype=np.float32)
    t = torch.ones(7, dtype=torch.float32)
    adapter = ActionAdapter()
    # preserve numpy
    out_np = adapter.canonical_to_dataset(arr, dataset_repo_id="libero")
    assert isinstance(out_np, np.ndarray)
    # preserve torch
    out_t = adapter.canonical_to_dataset(t, dataset_repo_id="libero")
    assert torch.is_tensor(out_t)
