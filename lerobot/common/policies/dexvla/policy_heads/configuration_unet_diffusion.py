#!/usr/bin/env python

# Copyright 2025 DexVLA Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Union

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class UnetDiffusionPolicyConfig(PretrainedConfig):
    """
    Configuration for dit diffusion policy head
    """

    model_type = "unet_diffusion_policy"

    def __init__(
        self,
        action_dim=10,
        global_cond_dim=2048,
        diffusion_step_embed_dim=256,
        down_dims=None,
        kernel_size=5,
        n_groups=8,
        state_dim=7,
        prediction_horizon=16,
        noise_samples=1,
        num_inference_timesteps=10,
        num_train_timesteps=100,
        **kwargs,
    ):
        if down_dims is None:
            down_dims = [256, 512, 1024]
        self.input_dim = action_dim
        self.noise_samples = noise_samples
        self.prediction_horizon = prediction_horizon
        self.num_inference_timesteps = num_inference_timesteps
        self.global_cond_dim = global_cond_dim
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.state_dim = state_dim
        self.num_train_timesteps = num_train_timesteps

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "llava_pythia":
            config_dict = config_dict["action_head"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
