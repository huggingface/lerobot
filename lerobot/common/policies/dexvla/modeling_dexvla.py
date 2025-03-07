from collections import deque

import torch
import torchvision.transforms as transforms
from torch import Tensor
from transformers import AutoProcessor, AutoTokenizer

from lerobot.common.policies.dexvla.configuration_dexvla import DexVLAConfig
from lerobot.common.policies.dexvla.qwe2_vla.modeling_qwen2_vla import Qwen2VLForConditionalGenerationForVLA
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

        self.model = Qwen2VLForConditionalGenerationForVLA(config.qwen2_vla_config).to(torch.bfloat16)
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
            reasonings = ["no reasoning"] * len(task_descs)

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
        image_grid_thw=None,
    ):
        input_ids = input_ids.to("cuda")
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
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

        if self.model.using_film:
            action_hidden_states = self.model.film_forward(
                labels=torch.ones_like(output_ids),
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
        image_grid_thw=None,
    ):
        input_ids = input_ids.to("cuda")
        with torch.inference_mode():
            all_hidden_states = self.model.forward(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
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
