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

from collections import deque

import torch
import torchvision.transforms as transforms
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from lerobot.common.policies.dexvla.configuration_dexvla import DexVLAConfig
from lerobot.common.policies.dexvla.robot_data_processor import Qwen2VLAProcess
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class DexVLAPolicy(PreTrainedPolicy):
    """Wrapper class around Qwen2VLForConditionalGenerationForVLA model to train and run inference within LeRobot."""

    config_class = DexVLAConfig
    name = "dexvla"

    def __init__(
        self,
        config: DexVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        for k in ["using_film", "llm_loss_weight", "with_llm_head", "policy_head_config"]:
            setattr(config.qwen2_vla_config, k, config.__dict__[k])

        # if self.config.training_stage == 2:
        # self.model = Qwen2VLForConditionalGenerationForVLA(config.qwen2_vla_config).to(torch.bfloat16)
        model_base = self.config.qwen2_vl_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_base,
            config=config.qwen2_vla_config,
            trust_remote_code=True,
            _fast_init=False,
            # attn_implementation="flash_attention_2",
        ).to(device="cuda", dtype=torch.bfloat16)

        if self.config.pretrained_scaledp_path is not None:
            print(
                "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Loading pretrained ScaleDP weights...<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
            pretrain_scaledp_weights = load_file(self.config.pretrained_scaledp_path)

            keys_to_del_dit = []
            pretrain_scaledp_weights = {
                k[7:] if k.startswith("policy.") else k: v for k, v in pretrain_scaledp_weights.items()
            }
            for k in pretrain_scaledp_weights:
                if "noise_pred" not in k:  # del weights of vision backbones
                    keys_to_del_dit.append(k)
                if "cond_obs_emb" in k:
                    keys_to_del_dit.append(k)
            for k in keys_to_del_dit:
                del pretrain_scaledp_weights[k]
            pretrain_scaledp_weights = {
                k[15:] if k.startswith("noise_pred_net.") else k: v
                for k, v in pretrain_scaledp_weights.items()
            }

            self.model.policy_head.load_state_dict(pretrain_scaledp_weights, strict=False)

        self.model.requires_grad_(False)
        self.model.policy_head.requires_grad_(True)
        self.qwen2_vl_processor = AutoProcessor.from_pretrained(config.qwen2_vl_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen2_vl_path)
        self.vla_processor = Qwen2VLAProcess(
            tokenizer=self.tokenizer, multimodal_processor=self.qwen2_vl_processor
        )  # process the input data into VLM format

        self.resize_size = self.config.resize_size
        ratio = 0.95
        self.transformations = [
            transforms.Resize(size=self.resize_size, antialias=True),
            transforms.RandomCrop(size=[int(self.resize_size[0] * ratio), int(self.resize_size[1] * ratio)]),
            transforms.Resize(self.resize_size, antialias=True),
            transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),  # , hue=0.08)
        ]

        self.reset()

    def process_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Applying DexVLA preprocessing to original data. Including resizing images. Scaling the range of actions, states."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        present_img_keys = [key for key in self.config.image_features if key in batch]
        task_descs = batch["task"]
        try:
            reasonings = batch["reasoning"]
        except KeyError:
            reasonings = ["None."] * len(task_descs)

        pass
        is_pad = batch["action_is_pad"]
        all_cam_images = []
        for k in present_img_keys:
            all_cam_images.append(batch[k])

        # construct observations, and scale 0-1 to 0-255
        image_data = torch.stack(all_cam_images) * 255
        image_data = image_data.to(dtype=torch.uint8)
        # construct observations
        qpos_data = batch["observation.state"].float()
        action_data = batch["action"].float()

        orig_shape = image_data.shape
        image_data = image_data.view(-1, *orig_shape[2:])

        for transform in self.transformations:
            image_data = transform(image_data)

        image_data = image_data.view(*orig_shape[:3], *self.resize_size)

        vl_data = {"images": image_data, "raw_langs": task_descs, "reasonings": reasonings}
        # processing vl_data into qwen2_vl format
        vla_inputs = self.vla_processor.forward(vl_data, use_reasoning=self.config.using_reasoning)
        vla_inputs["states"] = qpos_data
        vla_inputs["is_pad"] = is_pad
        vla_inputs["actions"] = action_data
        return vla_inputs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        processed_batch = self.process_batch(batch)

        ret = self.model.forward(**processed_batch)
        loss_dict = ret["loss"]
        loss = loss_dict["loss"].mean()
        return loss, loss_dict

    def dexvla_predict_action(
        self,
        input_ids: torch.LongTensor = None,
        actions=None,
        states=None,
        is_pad=None,
        tokenizer=None,
        is_eval=True,
        pixel_values=None,
        attention_mask=None,
        image_grid_spatiotemporal=None,
    ):
        input_ids = input_ids.to("cuda")
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_spatiotemporal=image_grid_spatiotemporal,
                is_eval=is_eval,
                num_beams=1,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=tokenizer.eos_token_id,  # Pad token
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        output_ids = outputs.sequences
        # last_hidden_states = outputs.hidden_states[-2][-1]
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        outputs_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]

        outputs_text = outputs_text.strip()
        last_hidden_states = [each[-1] for each in outputs.hidden_states]  # all hidden states
        all_hidden_states = torch.cat(last_hidden_states, dim=1)

        action_hidden_states = None
        labels_input = torch.ones((1, input_token_len)) * -100
        labels_output = torch.ones((1, output_ids.shape[1] - input_token_len))
        labels = torch.cat([labels_input, labels_output], dim=1)

        if self.model.using_film:
            action_hidden_states = self.model.film_forward(
                labels=labels,
                input_ids=output_ids,
                hidden_states=torch.cat(last_hidden_states, dim=1),
            )

        action = self.model.policy_head(
            actions, action_hidden_states, states.to(all_hidden_states.dtype), is_pad
        )
        return action, outputs_text

    def tinyvla_predict_action(
        self,
        input_ids: torch.LongTensor = None,
        actions=None,
        states=None,
        is_pad=None,
        is_eval=True,
        pixel_values=None,
        attention_mask=None,
        image_grid_spatiotemporal=None,
    ):
        input_ids = input_ids.to("cuda")
        with torch.inference_mode():
            all_hidden_states = self.model.forward(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_spatiotemporal=image_grid_spatiotemporal,
                is_eval=is_eval,
                tinyvla=True,
            )

        all_hidden_states = torch.mean(all_hidden_states, dim=1).unsqueeze(1)

        action = self.model.policy_head(
            actions, all_hidden_states, states.to(all_hidden_states.dtype), is_pad
        )
        return action, "tinyvla generates no reasoning"

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            present_img_keys = [key for key in self.config.image_features if key in batch]
            try:
                task_descs = batch["task"]
            except KeyError:
                task_descs = " "
                print("No task descriptions found for this task")

            all_cam_images = []
            for k in present_img_keys:
                all_cam_images.append(batch[k])

            # construct observations, and scale 0-1 to 0-255
            image_data = torch.stack(all_cam_images) * 255
            image_data = image_data.to(dtype=torch.uint8)
            # construct observations
            qpos_data = batch["observation.state"].float()

            image_data = image_data.squeeze(0)

            for transform in self.transformations:
                image_data = transform(image_data)

            # processing vl_data into qwen2_vl format
            vla_inputs = self.vla_processor.single_forward_process(
                images=image_data, raw_lang=task_descs, reasoning=None, eval=True
            )
            vla_inputs["states"] = qpos_data

            if self.config.using_film and self.config.with_llm_head:  # dexvla
                all_actions, outputs = self.dexvla_predict_action(
                    **vla_inputs, is_eval=True, tokenizer=self.tokenizer
                )
            else:  # tinyvla
                all_actions, outputs = self.tinyvla_predict_action(**vla_inputs, is_eval=True)

            actions = self.unnormalize_outputs({"action": all_actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()
