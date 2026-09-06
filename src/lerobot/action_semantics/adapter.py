from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conversion import to_canonical_action, from_canonical_action, convert_action
from .registry import (
    resolve_dataset_contract_from_repo_id,
    resolve_env_contract_from_env_type,
    get_dataset_action_contract,
)


@dataclass
class ActionTransition:
    policy_action: Any
    canonical_action: Any
    env_action: Any
    record_action: Any


class ActionAdapter:
    """Utility to convert between policy/dataset/env action representations.

    Uses the shared registry to resolve contracts by repo/env names.
    """

    def policy_to_canonical(self, policy_action, *, dataset_repo_id: str | None = None, policy_contract: Any | None = None, semantics: str = "per_step"):
        # If a dataset repo id is present, interpret the policy_action as matching
        # the dataset contract (legacy behavior). Otherwise, require explicit policy_contract.
        if dataset_repo_id is not None:
            source = resolve_dataset_contract_from_repo_id(dataset_repo_id)
        elif policy_contract is not None:
            source = policy_contract
        else:
            raise ValueError("Must supply dataset_repo_id or policy_contract to interpret policy_action")
        return to_canonical_action(policy_action, source, semantics=semantics)

    def canonical_to_environment(self, canonical_action, env_type: str | None = None, env_contract: Any | None = None, semantics: str = "per_step", clip: bool = False):
        if env_contract is None:
            if env_type is None:
                raise ValueError("Must supply env_type or env_contract")
            env_contract = resolve_env_contract_from_env_type(env_type)
        return from_canonical_action(canonical_action, env_contract, semantics=semantics, clip=clip)

    def canonical_to_dataset(self, canonical_action, dataset_repo_id: str | None = None, dataset_contract: Any | None = None, semantics: str = "per_step"):
        if dataset_contract is None:
            if dataset_repo_id is None:
                raise ValueError("Must supply dataset_repo_id or dataset_contract")
            dataset_contract = resolve_dataset_contract_from_repo_id(dataset_repo_id)
        return from_canonical_action(canonical_action, dataset_contract, semantics=semantics, clip=False)
