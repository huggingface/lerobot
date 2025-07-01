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

MODEL_STRUCTURE = {
    "scaledp_h": {
        "depth": 32,
        "n_emb": 1280,
        "num_heads": 16,
    },
    "scaledp_l": {
        "depth": 24,
        "n_emb": 1024,
        "num_heads": 16,
    },  # 400M
}


class ScaleDPPolicyConfig(PretrainedConfig):
    """
    Configuration for ScaleDP policy head
    """

    model_type = "scale_dp_policy"

    def __init__(
        self,
        eval: bool = False,
        action_dim: int = 14,  # action dim
        # output_dim: int = 14,  # action dim
        cond_dim: int = 1536,  # the input dim of the condition
        state_dim: int = 14,  # the input dim of the state
        prediction_horizon: int = 16,  # horizon
        n_obs_steps: int = 2,  # number of observation steps
        depth: int = 28,  # number of DiT blocks
        n_emb: int = 256,  # embedding size
        num_heads: int = 16,
        mlp_ratio: int = 4.0,
        time_as_cond: bool = True,
        obs_as_cond: bool = True,
        learn_sigma: bool = False,
        model_size: str = "none",
        num_inference_timesteps: int = 10,
        noise_samples: int = 1,
        num_train_timesteps: int = 100,
        **kwargs,
    ):
        if model_size != "none":
            depth = MODEL_STRUCTURE[model_size]["depth"]
            n_emb = MODEL_STRUCTURE[model_size]["n_emb"]
            num_heads = MODEL_STRUCTURE[model_size]["num_heads"]
        else:
            # raise ValueError("model_size show not be 'none'")
            pass
            # print("model_size should not be 'none'")
        self.eval = eval

        self.input_dim = action_dim
        self.output_dim = action_dim
        self.prediction_horizon = prediction_horizon

        self.cond_dim = cond_dim
        self.state_dim = state_dim

        self.n_obs_steps = n_obs_steps
        self.depth = depth
        self.n_emb = n_emb
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.learn_sigma = learn_sigma

        self.num_inference_timesteps = num_inference_timesteps
        self.num_queries = prediction_horizon
        self.noise_samples = noise_samples
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
