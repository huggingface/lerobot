import os
import json
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal

import ast
import torch.nn.functional as F

import logging
from .data_transform import (
    Normalizer,
    prepare_images,
    prepare_state,
    prepare_language,
    prepare_action,
    expert_visual_transform,
)
from .ee_pose_transform import *
from typing import Dict, List, Optional
import ast
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_image_token_count(images, image_grid_thw=None, merge_size=2, use_vision_boundaries=True):
    if not isinstance(images, torch.Tensor) or images.numel() == 0:
        return 0

    boundary_tokens = 2 if use_vision_boundaries else 0
    if image_grid_thw is not None:
        grid = image_grid_thw
        if not isinstance(grid, torch.Tensor):
            grid = torch.as_tensor(grid)
        grid = grid.reshape(-1, 3)
        num_patch = grid.prod(dim=-1) // (int(merge_size) ** 2)
        return int((num_patch + boundary_tokens).sum().item())

    if images.ndim == 3:
        num_images, num_patch = images.shape[0], images.shape[1]
    elif images.ndim >= 4:
        num_images, num_patch = images.shape[0], images.shape[-2]
    else:
        return 0
    return int(num_images * (num_patch + boundary_tokens))


class FeatureInfo(BaseModel):
    joints: List[str] | None = None
    images: List[str] | None = None
    joints_max_dim: dict | None = None

    def update_info(self, data_config):
        joints_info = data_config.joints
        self.images = ["observation.images." + image for image in data_config.cameras]

        joints = []
        joints_max_dim = {}
        for s in joints_info:
            joint_info = ast.literal_eval(s)
            joint = next(iter(joint_info.keys()))
            if joint_info[joint] == 0:
                continue

            joints.append(joint)
            joints_max_dim.update(joint_info)
        self.joints = joints
        self.joints_max_dim = joints_max_dim


class FeatureTransform:
    def __init__(
        self,
        robot_config_path,
        data_config,
        model_config,
        processor,
        disabled_image_features=False,
        do_nomalize=True,
        chunk_size=50,
        return_item_befor_padding=False,
        norm_stats_path=None,
        use_depth_align=False,
        image_augment=False,
        use_future_image=False,
    ):

        assert os.path.exists(robot_config_path), f"{robot_config_path} does not exist."
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        f.close()

        if norm_stats_path is None:
            norm_stats_path = robot_config.pop("norm_stats")
        else:
            robot_config.pop("norm_stats")

        self.feature_config = FeatureInfo()
        if getattr(data_config, "joints", None) is not None:
            self.feature_config.update_info(data_config)

        if not return_item_befor_padding:
            self.check_robot_config(robot_config)

        self.model_config = model_config
        self.tokenizer = processor.tokenizer if processor is not None else None
        self.processor = processor

        self.chunk_size = chunk_size
        self.return_item_befor_padding = return_item_befor_padding

        # disabled_image_features: keep the image features or not when getting lerobot item
        self.disabled_image_features = disabled_image_features
        self.use_depth_align = use_depth_align
        self.use_future_image = use_future_image

        if not disabled_image_features:
            self.image_augment = image_augment

        # keep the self.feature_to_keep in lerobot item when convert to new item
        self.feature_to_keep = set(
            [
                "timestamp",
                "frame_index",
                "episode_index",
                "task_index",
                "action_is_pad",
                "task",
            ]
        )

        target_features = {"states": [], "actions": [], "images": []}
        org_features = {"states": set(), "actions": set(), "images": set()}
        self.get_feature_mapping(robot_config, target_features, org_features)
        self.states = target_features["states"]
        self.actions = target_features["actions"]
        self.images = target_features["images"]

        self.org_features = org_features

        self.normalizer = self.get_normalizer(norm_stats_path, do_nomalize, data_config)

    def get_normalizer(self, norm_stats_path, do_nomalize, data_config):

        if not do_nomalize:
            return None

        action_state_norm_type = {k: v for d in data_config.norm_type for k, v in ast.literal_eval(d).items()}

        assert norm_stats_path is not None
        norm_type = {}
        for feature in self.actions:
            base_name = feature.split("action.")[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]

        for feature in self.states:
            base_name = feature.split("observation.state.")[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]

        for feature in self.images:
            norm_type[feature] = "identity"

        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        f.close()

        normalizer = Normalizer(
            norm_stats=norm_stats["norm_stats"],
            norm_type=norm_type,
        )

        return normalizer

    def check_robot_config(self, robot_config):

        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)
            for feature_convert_info in features_convert_info:
                if isinstance(feature_convert_info, dict):
                    assert len(feature_convert_info.keys()) == 1

                if feature_category == "actions":
                    assert isinstance(feature_convert_info, dict)
                    action_feature = list(feature_convert_info.keys())[0].split("action.")[-1]
                    if action_feature not in self.feature_config.joints:
                        raise ValueError(
                            f"{action_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}"
                        )

                elif feature_category == "states":
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        state_feature = list(feature_convert_info.keys())[0]
                    else:
                        state_feature = feature_convert_info
                    state_feature = state_feature.split("observation.state.")[-1]
                    if state_feature not in self.feature_config.joints:
                        raise ValueError(
                            f"{state_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}"
                        )

                elif feature_category == "images":
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        image_feature = list(feature_convert_info.keys())[0]
                    else:
                        image_feature = feature_convert_info
                    if image_feature not in self.feature_config.images:
                        raise ValueError(
                            f"{image_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.images}"
                        )

    def get_feature_mapping(self, robot_config, target_features, org_features):
        to_convert_features = {}
        reverse_convert_features = {}
        action_subtract_state = {}
        action_relative_type = {}
        actions_convert_from_state = set()
        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)

            for feature_convert_info in features_convert_info:
                if isinstance(feature_convert_info, str):
                    self.feature_to_keep.add(feature_convert_info)
                    target_features[feature_category].append(feature_convert_info)
                    org_features[feature_category].add(feature_convert_info)

                elif isinstance(feature_convert_info, dict):
                    target_feature = next(iter(feature_convert_info.keys()))
                    target_features[feature_category].append(target_feature)

                    if feature_category == "actions":
                        assert isinstance(feature_convert_info, dict)
                        assert "subtract_state" in feature_convert_info[target_feature]
                        action_subtract_state[target_feature] = feature_convert_info[target_feature].pop(
                            "subtract_state"
                        )
                        if "velocity" in target_feature or "effector.position" in target_feature:
                            assert not action_subtract_state[target_feature], (
                                f"{target_feature} cannot be subtracted from state"
                            )

                        action_relative_type[target_feature] = feature_convert_info[target_feature].pop(
                            "relative_type", "subtract"
                        )
                        if_convert_from_state = feature_convert_info[target_feature].pop(
                            "convert_from_state", False
                        )
                        if if_convert_from_state:
                            actions_convert_from_state.add(target_feature)

                        if (
                            "origin_keys" not in feature_convert_info[target_feature]
                            and not if_convert_from_state
                        ):
                            self.feature_to_keep.add(feature_convert_info)

                    if "origin_keys" in feature_convert_info[target_feature]:
                        if isinstance(feature_convert_info[target_feature]["origin_keys"], list):
                            ordered_origin_keys = OrderedDict()
                            for item in feature_convert_info[target_feature]["origin_keys"]:
                                for k, v in item.items():
                                    while k in ordered_origin_keys:
                                        k = k + "*"
                                    ordered_origin_keys[k] = v

                            feature_convert_info[target_feature]["origin_keys"] = ordered_origin_keys
                            to_convert_features.update(feature_convert_info)

                            if feature_category in ["actions", "states"]:
                                org_features[feature_category].update(
                                    [k.split("*")[0] for k in ordered_origin_keys.keys()]
                                )

                            target_start_id = 0
                            for org_key, info in ordered_origin_keys.items():
                                org_info = info.copy()
                                org_key = org_key.split("*")[0]
                                if org_key not in reverse_convert_features:
                                    reverse_convert_features[org_key] = []

                                if "start" in org_info:
                                    org_info["target_key"] = target_feature
                                    org_info["target_start"] = target_start_id
                                    org_info["target_end"] = (
                                        target_start_id + org_info["end"] - org_info["start"]
                                    )
                                    target_start_id = org_info["target_end"]
                                else:
                                    org_info["target_key"] = target_feature
                                reverse_convert_features[org_key].append(org_info)

                        if isinstance(feature_convert_info[target_feature]["origin_keys"], str):
                            to_convert_features[target_feature] = feature_convert_info[target_feature]
                            reverse_convert_features[feature_convert_info[target_feature]["origin_keys"]] = {
                                "target_key": target_feature
                            }
                            org_features[feature_category].add(
                                feature_convert_info[target_feature]["origin_keys"]
                            )

        to_convert_state = {}
        if len(actions_convert_from_state) > 0:
            for action in list(actions_convert_from_state):
                assert action.replace("action.", "observation.state.") in target_features["states"], (
                    f"{action} can not converted from {action.replace('action.', 'observation.state.')}"
                )
                to_convert_state[action] = action.replace("action.", "observation.state.")
        self.actions_convert_from_state = to_convert_state

        if len(self.actions_convert_from_state) > 0:
            # To keep time alignment, if any action is converted from state, all actions must be converted from state
            assert len(self.actions_convert_from_state) == len(target_features["actions"])

        self.action_subtract_state = action_subtract_state
        self.action_relative_type = action_relative_type
        for feature_category, feature in org_features.items():
            if feature_category == "actions":
                if len(feature) == 0 and len(self.actions_convert_from_state) > 0:
                    org_features["actions"] == []
                    continue
            if len(feature) == 0:
                org_features[feature_category] = target_features[feature_category]
            else:
                org_features[feature_category] = list(feature)

        self.key_mapping = to_convert_features
        self.key_reverse_mapping = reverse_convert_features

    def convert_features(self, item, w_action):
        out_item = {}
        for target_key, convert_info in self.key_mapping.items():
            if self.disabled_image_features and target_key in self.images:
                continue
            if not w_action and "action" in target_key:
                continue
            if isinstance(convert_info["origin_keys"], str) and convert_info["origin_keys"] in item:
                out_item[target_key] = item[convert_info["origin_keys"]]
                continue

            assert isinstance(convert_info["origin_keys"], OrderedDict)
            concat_list = []
            convert_success = True
            for origin_key, origin_info in convert_info["origin_keys"].items():
                origin_key = origin_key.split("*")[0]
                if origin_key not in item:
                    convert_success = False
                    break
                origin_data = item.get(origin_key)[..., origin_info["start"] : origin_info["end"]]
                concat_list.append(origin_data)
            if convert_success:
                out_item[target_key] = torch.cat(concat_list, dim=-1)
            del concat_list

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]

        if len(self.actions_convert_from_state) > 0 and w_action:
            for action_feature in self.actions:
                state_feature = self.actions_convert_from_state[action_feature]
                assert state_feature in out_item and len(out_item[state_feature].shape) == 2
                out_item[action_feature] = out_item[state_feature][1:].clone()
                out_item[state_feature] = out_item[state_feature][0].clone()
            for state_feature in self.states:
                if len(out_item[state_feature].shape) == 2:
                    out_item[state_feature] = out_item[state_feature][0]

        del item
        return out_item

    def reverse_features(self, item):
        if len(self.actions_convert_from_state) > 0:
            for action_feature, state_feature in self.actions_convert_from_state.items():
                item[state_feature] = torch.cat(
                    [item[state_feature].unsqueeze(0), item[action_feature]], dim=0
                )

                item.pop(action_feature)

        out_item = {}

        for target_key, convert_info in self.key_reverse_mapping.items():
            if isinstance(convert_info, dict) and convert_info["target_key"] in item:
                out_item[target_key] = item[convert_info["target_key"]]
                continue

            if isinstance(convert_info, list):
                convert_info = sorted(convert_info, key=lambda x: x["end"])
                concat_list = []
                convert_success = True
                for _convert_info in convert_info:
                    if _convert_info["target_key"] not in item:
                        raise ValueError(
                            f"{_convert_info['target_key']} is not contained in robot config as target feature"
                        )
                    concat_list.append(
                        item[_convert_info["target_key"]][
                            ..., _convert_info["target_start"] : _convert_info["target_end"]
                        ]
                    )
                if convert_success:
                    out_item[target_key] = torch.cat(concat_list, dim=-1)

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]
        del item
        return out_item

    def apply(self, item, policy_eval=False):
        w_action = not policy_eval
        if w_action:
            item["action_is_pad"] = (
                item[f"{self.org_features['actions'][0]}_is_pad"]
                if not len(self.actions_convert_from_state) > 0
                else item[f"{self.org_features['states'][0]}_is_pad"][1:]
            )
        else:
            item["action_is_pad"] = torch.zeros(self.chunk_size)
        item = self.convert_features(item, w_action=w_action)

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature] and w_action:
                state_feature = action_feature.replace("action.", "observation.state.")
                if not (action_feature in item and state_feature in item):
                    raise ValueError(f"{action_feature} or/and {state_feature} are not in the item")
                relative_type = self.action_relative_type.get(action_feature)
                if _is_quaternion_relative_type(relative_type):
                    assert "end.position" in action_feature
                    item[action_feature] = relative_pose_quaternion(
                        item[action_feature],
                        item[state_feature],
                        relative_type=relative_type,
                    )
                else:
                    item[action_feature] -= item[state_feature]

        if self.normalizer is not None:
            item = self.normalizer.normalize(item)

        if self.return_item_befor_padding:
            return item

        batch_dict = self.pad_and_concat(item, w_action)

        state = prepare_state(batch_dict, self.model_config.max_state_dim)
        actions = prepare_action(batch_dict, self.model_config.max_action_dim)
        return_image_grid_thw = getattr(self.model_config, "return_image_grid_thw", False)
        if not self.disabled_image_features:
            (
                images,
                img_masks,
                pil_images,
                image_grid_thw,
                image_augment_params,
            ) = prepare_images(
                self.processor.image_processor,
                batch_dict,
                image_keys=self.feature_config.images,
                train=self.image_augment and not policy_eval,
                use_depth_align=self.use_depth_align,
                return_image_grid_thw=return_image_grid_thw,
                return_augment_params=True,
            )
            if self.use_future_image and len(batch_dict.get("future_image", {})) > 0:
                future_obs = {**batch_dict, "image": batch_dict["future_image"]}
                future_images, _, future_pil_images, _ = prepare_images(
                    self.processor.image_processor,
                    future_obs,
                    image_keys=self.feature_config.images,
                    train=self.image_augment and not policy_eval,
                    use_depth_align=self.use_depth_align,
                    return_image_grid_thw=False,
                    augment_params=image_augment_params,
                )
            else:
                future_images, future_pil_images = None, None
        else:
            images, img_masks, pil_images, image_grid_thw = [], [], [], None
            future_images, future_pil_images = None, None

        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        image_token_count = compute_image_token_count(
            images,
            image_grid_thw=image_grid_thw,
            merge_size=merge_size,
            use_vision_boundaries=getattr(self.model_config, "qwen3vl_use_vision_boundaries", True),
        )
        batch_dict["image_token_count"] = image_token_count

        lang_tokens, lang_masks = prepare_language(
            self.model_config, self.tokenizer, batch_dict
        )  # bs, seq_len
        action_is_pad = batch_dict["action_is_pad"]

        state_joint_mask = batch_dict["state_joint_mask"]
        assert self.model_config.max_state_dim >= state_joint_mask.shape[-1], (
            f"max_action_dim is smaller than the state joint dimension: {self.model_config.max_action_dim} < {state_joint_mask.shape[-1]}"
        )
        state_joint_mask = F.pad(
            state_joint_mask, (0, self.model_config.max_state_dim - state_joint_mask.shape[-1])
        ).to(dtype=torch.bool)

        action_joint_mask = batch_dict["action_joint_mask"]
        assert self.model_config.max_action_dim >= action_joint_mask.shape[-1], (
            f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {action_joint_mask.shape[-1]}"
        )
        action_joint_mask = F.pad(
            action_joint_mask, (0, self.model_config.max_action_dim - action_joint_mask.shape[-1])
        ).to(dtype=torch.bool)

        chunk_joint_mask = batch_dict["chunk_joint_mask"]
        assert self.model_config.max_action_dim >= chunk_joint_mask.shape[-1], (
            f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {chunk_joint_mask.shape[-1]}"
        )
        chunk_joint_mask = F.pad(
            chunk_joint_mask, (0, self.model_config.max_action_dim - chunk_joint_mask.shape[-1])
        ).to(dtype=torch.bool)

        del batch_dict
        batch_dict = {
            "images": images,
            "img_masks": img_masks,
            "state": state,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "joint_mask": chunk_joint_mask,
            "state_joint_mask": state_joint_mask,
            "action_joint_mask": action_joint_mask,
        }
        if image_grid_thw is not None:
            batch_dict["image_grid_thw"] = image_grid_thw

        if self.use_depth_align:
            batch_dict["pil_images"] = pil_images
            if self.use_future_image:
                # assert future_pil_images is not None and future_pil_images is not []:
                batch_dict["future_pil_images"] = future_pil_images
        return batch_dict

    def unapply(self, item):
        if not self.return_item_befor_padding:
            item = self.reverse_pad_and_concat(item)

        if self.normalizer is not None:
            item = self.normalizer.unnormalize(item)

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature]:
                state_feature = action_feature.replace("action.", "observation.state.")
                relative_type = self.action_relative_type.get(action_feature)
                if _is_quaternion_relative_type(relative_type):
                    assert "end.position" in action_feature
                    item[action_feature] = absolute_pose_quaternion(
                        item[action_feature],
                        item[state_feature],
                        relative_type=relative_type,
                    )
                else:
                    item[action_feature] += item[state_feature]

        item = self.reverse_features(item)
        return item

    def reverse_pad_and_concat(self, item):
        reverse_item = {}

        # In policy_eval, model output `actions` is always padded to max_action_dim
        # Pad mask with False at the tail so `actions[:, mask]` selects only the
        # real joint dims — the padded region is False and contributes nothing.
        state_joint_mask = item["state_joint_mask"]
        assert state_joint_mask.shape[-1] == item["state"].shape[-1]

        action_joint_mask = item["action_joint_mask"]
        assert action_joint_mask.shape[-1] == item["actions"].shape[-1]

        state = item["state"][state_joint_mask]
        action = item["actions"][:, action_joint_mask]

        for k in self.feature_config.joints:
            state_key = f"observation.state.{k}"
            if state_key in self.states:
                joint_dim = self.normalizer.norm_stats[state_key]["mean"].shape[-1]
                reverse_item[state_key] = state[:joint_dim]
                state = state[joint_dim:]
            del state_key

            action_key = f"action.{k}"
            if action_key in self.actions:
                joint_dim = self.normalizer.norm_stats[action_key]["mean"].shape[-1]
                reverse_item[action_key] = action[:, :joint_dim]
                action = action[:, joint_dim:]
            del action_key
        return reverse_item

    def pad_and_concat(self, item, w_action=True):
        images = {}
        future_images = {}
        for image_key in self.feature_config.images:
            if image_key in self.images and image_key in item:
                if not self.use_future_image:
                    images[image_key] = item[image_key]
                else:
                    images[image_key] = item[image_key][0]
                    future_images[image_key] = item[image_key][-1]

        actions, action_joints_pad = [], []
        states, state_joints_pad = [], []

        for k in self.feature_config.joints:
            state_key = f"observation.state.{k}"

            if state_key in self.states:
                pad_len = self.feature_config.joints_max_dim[k] - item[state_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                states.append(F.pad(item[state_key], (0, pad_len)))
                state_joints_pad.append(F.pad(torch.ones(item[state_key].shape), (0, pad_len)))
            else:
                states.append(torch.zeros(self.feature_config.joints_max_dim[k]))
                state_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del state_key

            action_key = f"action.{k}"
            if action_key in self.actions and w_action:
                pad_len = self.feature_config.joints_max_dim[k] - item[action_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                actions.append(F.pad(item[action_key], (0, pad_len)))
                action_joints_pad.append(F.pad(torch.ones(item[action_key].shape[-1]), (0, pad_len)))
            elif action_key in self.actions and not w_action:
                assert action_key in self.normalizer.norm_stats, (
                    f"{action_key} not in norm keys: {self.normalizer.norm_stats.keys()}"
                )
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                pad_len = (
                    self.feature_config.joints_max_dim[k]
                    - self.normalizer.norm_stats[action_key]["mean"].shape[-1]
                )
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                action_joints_pad.append(
                    F.pad(torch.ones(self.normalizer.norm_stats[action_key]["mean"].shape[-1]), (0, pad_len))
                )
            else:
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                action_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del action_key

        action_joint_mask = torch.cat(action_joints_pad, dim=-1).to(dtype=torch.bool)
        state_joint_mask = torch.cat(state_joints_pad, dim=-1).to(dtype=torch.bool)
        state = torch.cat(states, dim=-1).to(torch.float32)
        action = torch.cat(actions, dim=-1).to(torch.float32)
        chunk_joint_mask = action_joint_mask.clone().unsqueeze(0).repeat(self.chunk_size, 1)

        batch_dict = {
            "image": images,
            "future_image": future_images,
            "state": state,
            "action": action,
            "action_is_pad": item["action_is_pad"],
            "chunk_joint_mask": chunk_joint_mask,
            "action_joint_mask": action_joint_mask,
            "state_joint_mask": state_joint_mask,
            "prompt": [item["task"]],
        }
        if "future_video_effective_fps" in item:
            batch_dict["future_video_effective_fps"] = item["future_video_effective_fps"]

        return batch_dict
