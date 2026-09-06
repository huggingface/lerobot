"""Registry for dataset/env/policy ActionContracts.

This module exposes helpers to resolve dataset, environment, and policy-level
action contracts. Policies should prefer declaring their action semantics via
their `PreTrainedConfig` rather than relying on implicit defaults.
"""

from __future__ import annotations

from .contracts import (
    ActionContract,
    LIBERO_ACTION_CONTRACT,
    LIBERO_DATASET_ACTION_CONTRACT,
    LIBERO_ENV_ACTION_CONTRACT,
    LIBERO_SAFETY_DATASET_ACTION_CONTRACT,
    LIBERO_SAFETY_ENV_ACTION_CONTRACT,
)


_DATASET_MAP: dict[str, ActionContract] = {
    "libero": LIBERO_DATASET_ACTION_CONTRACT,
    "libero_safety": LIBERO_SAFETY_DATASET_ACTION_CONTRACT,
}

_ENV_MAP: dict[str, ActionContract] = {
    "libero": LIBERO_ENV_ACTION_CONTRACT,
    "libero_safety": LIBERO_SAFETY_ENV_ACTION_CONTRACT,
}


# Policy-level defaults: policies should declare their contract where possible. We provide
# conservative defaults for known policies here to avoid breaking legacy models.
_POLICY_MAP: dict[str, ActionContract] = {
    "safediff_vla": LIBERO_ACTION_CONTRACT,
    "act": LIBERO_ACTION_CONTRACT,
    "diffusion": LIBERO_ACTION_CONTRACT,
    "smolvla": LIBERO_ACTION_CONTRACT,
    "pi0": LIBERO_ACTION_CONTRACT,
    "vla_jepa": LIBERO_ACTION_CONTRACT,
}


def get_dataset_action_contract(name: str) -> ActionContract:
    """Return a dataset-level ActionContract by key.

    Raises KeyError for unknown names.
    """
    return _DATASET_MAP[name]


def get_env_action_contract(name: str) -> ActionContract:
    """Return an environment-level ActionContract by key.

    Raises KeyError for unknown names.
    """
    return _ENV_MAP[name]


def get_policy_action_contract(policy: object | str) -> ActionContract:
    """Resolve a policy-level ActionContract.

    Accepts either a string policy name, a PreTrainedConfig-like object with attribute
    ``name`` or ``model_type``, or a policy instance with ``config`` attribute. When the
    policy is unknown, this function raises a ValueError to force callers to be explicit.
    """
    # string name
    if isinstance(policy, str):
        key = policy.lower()
        if key in _POLICY_MAP:
            return _POLICY_MAP[key]
        raise ValueError(f"Unknown policy name '{policy}'. Register its action contract in the policy map.")

    # policy instance with config
    cfg = getattr(policy, "config", None)
    if cfg is not None:
        # first, explicit named contract
        pac = getattr(cfg, "policy_action_contract", None)
        if isinstance(pac, str) and pac:
            key = pac.lower()
            if key in _POLICY_MAP:
                return _POLICY_MAP[key]
            if key in _DATASET_MAP:
                return _DATASET_MAP[key]
            if key in _ENV_MAP:
                return _ENV_MAP[key]
            raise ValueError(f"policy_action_contract '{pac}' is not recognized; register it first")

        # next, declarative representation + optional domain
        rep = getattr(cfg, "action_representation", None)
        domain = getattr(cfg, "action_domain", None)
        if isinstance(rep, str) and rep:
            rep = rep
            # if domain is provided, try to copy controller scales / fps from known dataset/env
            domain_contract = None
            if isinstance(domain, str) and domain.lower() in _DATASET_MAP:
                domain_contract = _DATASET_MAP[domain.lower()]
            elif isinstance(domain, str) and domain.lower() in _ENV_MAP:
                domain_contract = _ENV_MAP[domain.lower()]

            if domain_contract is not None:
                return ActionContract(
                    name=f"{getattr(cfg, 'type', 'policy')}_policy",
                    action_dim=domain_contract.action_dim,
                    fps=domain_contract.fps,
                    representation=rep,  # controller_command or physical_delta/velocity
                    controller_translation_scale=domain_contract.controller_translation_scale,
                    controller_rotation_scale=domain_contract.controller_rotation_scale,
                )

            # No domain: only allow non-controller representations without silent assumptions
            if rep == "controller_command":
                raise ValueError("action_domain is required when action_representation='controller_command' to infer controller scales")
            # create a minimal ActionContract with the declared representation
            return ActionContract(name=f"{getattr(cfg, 'type', 'policy')}_policy", representation=rep)

        # try known policy map by type/name
        key = getattr(cfg, "name", None) or getattr(cfg, "model_type", None) or getattr(cfg, "type", None)
        if isinstance(key, str) and key.lower() in _POLICY_MAP:
            return _POLICY_MAP[key.lower()]

    raise ValueError("Could not resolve policy ActionContract; please register it or provide a policy config with an explicit action contract or domain.")


def resolve_dataset_contract_from_repo_id(repo_id: str) -> ActionContract:
    """Resolve a dataset contract from a repo id string by simple heuristics.

    Keeps logic small: looks for 'libero_safety' substring first, then 'libero'.
    """
    key = repo_id.lower()
    if "libero_safety" in key or "libero-safety" in key:
        return _DATASET_MAP["libero_safety"]
    if "libero" in key:
        return _DATASET_MAP["libero"]
    # fallback: return libero dataset contract
    return _DATASET_MAP["libero"]


def resolve_env_contract_from_env_type(env_type: str) -> ActionContract:
    """Resolve an env contract heuristically from an env type string."""
    key = env_type.lower()
    if "libero_safety" in key or "libero-safety" in key:
        return _ENV_MAP["libero_safety"]
    if "libero" in key:
        return _ENV_MAP["libero"]
    return _ENV_MAP["libero"]
