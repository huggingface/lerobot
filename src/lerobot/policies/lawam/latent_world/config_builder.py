from __future__ import annotations

from dataclasses import fields
from typing import Any

from lerobot.policies.lawam.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from lerobot.policies.lawam.vlas.lawam import LatentWorldPolicyConfig


class LatentWorldPolicyConfigBuilder:
    """Build LatentWorldPolicyConfig from global training config with strict field validation."""

    _DEPRECATED_FIELDS = {"model_id", "yaml_path"}
    _DATASET_FIELDS = {"enable_primary_video_aug", "enable_primary_random_resized_crop"}
    _WINDOW_KEYS = {"future_action_window_size", "past_action_window_size", "action_horizon"}
    _MIGRATED_FREEZE_FIELDS = {
        "freeze_vision_backbone",
        "freeze_llm_backbone",
        "freeze_last_llm_layer",
        "freeze_embedding",
        "unfreeze_vision_merger",
        "keep_llm_first_n_layers",
        "unfreeze_llm_last_n_layers",
        "unfreeze_lam_decoder",
    }

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def build(self) -> LatentWorldPolicyConfig:
        policy_cfg = LatentWorldPolicyConfig()
        framework_cfg = self.cfg.framework
        if "latent_world" in framework_cfg:
            raise ValueError(
                "`framework.latent_world` is no longer supported. "
                "Use `framework.action_model` with required keys "
                "`future_action_window_size`, `past_action_window_size`, and `action_horizon`."
            )
        if "action_model" not in framework_cfg:
            raise ValueError(
                "Missing `framework.action_model` for LaWAM. "
                "Please configure LatentWorld parameters under `framework.action_model`."
            )
        am = framework_cfg.action_model

        missing_window_keys = sorted(k for k in self._WINDOW_KEYS if am.get(k, None) is None)
        if missing_window_keys:
            raise ValueError(
                f"Missing required action chunk fields in `framework.action_model`: {missing_window_keys}."
            )
        future_window = int(am.future_action_window_size)
        past_window = int(am.past_action_window_size)
        action_horizon = int(am.action_horizon)
        expected_action_horizon = int(future_window + past_window + 1)
        if action_horizon != expected_action_horizon:
            raise ValueError(
                "Invalid action chunk config: expected "
                "`action_horizon = future_action_window_size + past_action_window_size + 1`, got "
                f"action_horizon={action_horizon}, future_action_window_size={future_window}, "
                f"past_action_window_size={past_window}."
            )

        top_keys = set(am.keys())
        deprecated = sorted(self._DEPRECATED_FIELDS.intersection(top_keys))
        if deprecated:
            raise ValueError(
                "Deprecated action_model fields are not allowed for LaWAM: "
                f"{deprecated}. Use `framework.qwenvl.base_vlm` as VLM source."
            )

        migrated_freeze = sorted(self._MIGRATED_FREEZE_FIELDS.intersection(top_keys))
        if migrated_freeze:
            raise ValueError(
                "LatentWorld freeze fields in `framework.action_model` are not allowed. "
                "Use `trainer.freeze`: "
                f"{migrated_freeze}."
            )

        misplaced_dataset_fields = sorted(self._DATASET_FIELDS.intersection(top_keys))
        if misplaced_dataset_fields:
            raise ValueError(
                "Dataset-side LatentWorld fields are not allowed in `framework.action_model`. "
                "Move them to `datasets.vla_data`: "
                f"{misplaced_dataset_fields}."
            )

        valid_top_keys = {f.name for f in fields(LatentWorldPolicyConfig)} - {"flow_cfg"}
        unknown_top = sorted(
            k for k in top_keys if k not in valid_top_keys and k not in self._WINDOW_KEYS and k != "flow_cfg"
        )
        if unknown_top:
            raise ValueError(f"Unknown `framework.action_model` fields for LatentWorld: {unknown_top}")

        for key in top_keys:
            if key != "flow_cfg" and key not in self._WINDOW_KEYS:
                setattr(policy_cfg, key, am[key])

        policy_cfg.future_action_window_size = future_window
        policy_cfg.past_action_window_size = past_window
        policy_cfg.action_horizon = action_horizon

        flow_cfg = am.get("flow_cfg", None)
        if "flow_cfg" in top_keys and flow_cfg is not None:
            flow_keys = set(flow_cfg.keys())
            if "window_size" in flow_keys:
                raise ValueError(
                    "`framework.action_model.flow_cfg.window_size` has been removed. "
                    "Configure only `framework.action_model.action_horizon`."
                )
            legacy_flow_keys = sorted(k for k in ("proprio_dim", "use_proprio") if k in flow_keys)
            if legacy_flow_keys:
                raise ValueError(
                    "Legacy flow_cfg fields are not supported: "
                    f"{legacy_flow_keys}. Use `state_dim` / `use_state`."
                )
            deprecated_flow_keys = sorted(k for k in ("state_dropout_prob",) if k in flow_keys)
            if deprecated_flow_keys:
                raise ValueError(
                    "Deprecated flow_cfg fields are not supported: "
                    f"{deprecated_flow_keys}. Use `use_state` to enable or disable state conditioning."
                )
            valid_flow_keys = {f.name for f in fields(ConditionalFlowMatchingConfig)}
            unknown_flow = sorted(k for k in flow_keys if k not in valid_flow_keys)
            if unknown_flow:
                raise ValueError(f"Unknown `framework.action_model.flow_cfg` fields: {unknown_flow}")
            for key in flow_keys:
                setattr(policy_cfg.flow_cfg, key, flow_cfg[key])

        return policy_cfg
