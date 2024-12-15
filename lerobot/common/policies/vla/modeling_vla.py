from collections import deque

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from torch.profiler import record_function
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import einops

from lerobot.common.policies.act.modeling_act import ACTDecoder, ACTEncoderDecoder
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.vla.configuration_vla import VLAConfig


class VLAPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "act"],
):
    """
    Vision-Language Action Policy (VLAPolicy).
    This policy uses a Vision-Language Model (VLA) for action prediction based on vision and language inputs.
    """

    name = "vla"

    def __init__(
        self,
        config: VLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Initialize the VLAPolicy class with model configuration.

        Args:
        config (VLAConfig): Configuration for the Qwen2VL model.
        """
        super().__init__()
        if config is None:
            config = VLAConfig()
        self.config: VLAConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VLA(config)
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()

        with record_function("normalize_inputs"):
            batch = self.normalize_inputs(batch)

        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = [img for k in self.expected_image_keys for img in batch[k]]
        batch["prompt"] = self.config.prompt
        # Forward pass through VLA
        with record_function("model"):
            predicted_actions = self.model(batch)
        with record_function("unnormalize_outputs"):
            if len(self._action_queue) == 0:
                actions = self.unnormalize_outputs({"action": predicted_actions})["action"]
                self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)
            batch["observation.images"] = [img for k in self.expected_image_keys for img in batch[k]]
        batch["prompt"] = self.config.prompt

        predicted_actions = self.model(batch)

        loss_dict = {}
        if "action" in batch:
            true_actions = batch["action"]
            l1_loss = (
                F.l1_loss(predicted_actions, true_actions, reduction="none")
                * ~batch["action_is_pad"].unsqueeze(-1)
            ).mean()
            loss_dict["l1_loss"] = l1_loss.item()
            loss_dict["loss"] = l1_loss

        return loss_dict


class VLA(nn.Module):
    def __init__(self, config: VLAConfig, device: torch.device = "cpu"):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.action_decoder_name = config.action_decoder.get("name", "act")
        self.use_robot_state = "observation.state" in config.input_shapes
        if self.action_decoder_name == "act_decoder":
            action_decoder_config = OmegaConf.create(config.action_decoder)
            self.action_decoder = ACTDecoder(action_decoder_config)
            self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.action_decoder["dim_model"])
        elif self.action_decoder_name == "act":
            action_decoder_config = OmegaConf.create(config.action_decoder)
            self.action_decoder = ACTEncoderDecoder(action_decoder_config)
        else:
            raise NotImplementedError(f"{self.action_decoder_name} not supported.")

        self.action_head = nn.Linear(config.action_decoder["dim_model"], config.output_shapes["action"][0])

        self.vlm_backbone_name = config.vlm_backbone["name"]
        self.vlm_backbone_feature_selection = config.vlm_backbone.get("feature_selection", "last_token")
        self.vlm_hidden_size = config.vlm_backbone.get("hidden_size", config.action_decoder["dim_model"])
        if "llava-onevision" in self.vlm_backbone_name:
            self.vision_language_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.vlm_backbone_name,
                device_map="cuda",
                #torch_dtype=torch.float16,
                #attn_implementation="flash_attention_2"
            )
            self.processor = AutoProcessor.from_pretrained(self.vlm_backbone_name)
        elif "paligemma" in self.vlm_backbone_name:
            from transformers import PaliGemmaForConditionalGeneration

            self.vision_language_model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.vlm_backbone_name,
                device_map="cuda",
                # torch_dtype=torch.float16,
                # attn_implementation="flash_attention_2"
            )
            self.processor = AutoProcessor.from_pretrained(self.vlm_backbone_name)
        elif "SmolVLM" in self.vlm_backbone_name:
            from transformers import Idefics3ForConditionalGeneration
            self.vision_language_model = Idefics3ForConditionalGeneration.from_pretrained(
                self.vlm_backbone_name,
                device_map="cuda",
                #torch_dtype=torch.float16,
                #attn_implementation="flash_attention_2"
            )
            self.processor = AutoProcessor.from_pretrained(self.vlm_backbone_name)
            self.processor.image_processor.do_image_splitting = False
        elif "resnet" in self.vlm_backbone_name:
            import torchvision
            from torchvision.models._utils import IntermediateLayerGetter
            from torchvision.ops.misc import FrozenBatchNorm2d
            vision_backbone = "resnet18"
            pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1"
            replace_final_stride_with_dilation = False
            backbone_model = getattr(torchvision.models, vision_backbone)(
                replace_stride_with_dilation=[False, False, replace_final_stride_with_dilation],
                weights=pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.vision_language_model = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        else:
            raise NotImplementedError(f"{self.vlm_backbone_name} not supported.")
        self.use_prompt_template = config.use_prompt_template
        self.num_img_tokens = config.num_img_tokens  #  e.g. 598 to match the number of hidden states in ACT

        if self.use_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], self.vlm_hidden_size
            )

        self.use_action_connector = config.use_action_connector
        if self.use_action_connector:
            self.action_connector = nn.Linear(self.vlm_hidden_size, config.action_decoder["dim_model"])

        self.peft_method = config.peft_method
        if "lora" in self.peft_method:
            peft_config = config.peft_config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # Based on the task type (e.g., language modeling, etc.)
                r=peft_config["r"],  # The rank of the low-rank adaptation
                lora_alpha=peft_config["lora_alpha"],  # Scaling factor
                lora_dropout=peft_config["lora_dropout"],  # Dropout applied to LoRA layers
                target_modules=peft_config["target_modules"],  # The components where LoRA is applied
            )
            self.lora_config = lora_config
            # Apply LoRA and ensure only LoRA parameters are trainable
            self.vision_language_model = get_peft_model(self.vision_language_model, lora_config)
            for name, param in self.vision_language_model.named_parameters():
                if (
                    "lm_head" in name or "lora" in name
                ):  # lm_head is not a parameter in most LLMs becasue it's tied to the embedding layer
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif "freeze" in self.peft_method:
            for name, param in self.vision_language_model.named_parameters():
                param.requires_grad = False

        # Verify trainable parameters
        trainable_params = []
        for name, param in self.vision_language_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
                print(f"VLM trainable parameter: {name}")

    def apply_prompt_template(self, text: str, add_generation_prompt: bool = True) -> str:
        if "llava-onevision" in self.vlm_backbone_name:
            if  self.use_prompt_template:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image"},
                        ],
                    },
                ]
                prompt = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=add_generation_prompt
                )
            else:
                prompt = f"<image>{text}"
        elif "paligemma" in self.vlm_backbone_name:
            prompt = f"<image>{text}"
        elif "SmolVLM" in self.vlm_backbone_name:
            conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"}
                ]
            },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=add_generation_prompt
            )
        else:
            prompt = text
        return prompt

    def get_vlm_features(self, processed_inputs) -> torch.Tensor:

        vlm_output = self.vision_language_model(
            **processed_inputs, return_dict=True, output_hidden_states=True
        )

        if any(k in self.vlm_backbone_name for k in ["llava-onevision", "paligemma", "SmolVLM"]):
            if "llava-onevision" in self.vlm_backbone_name:
                batch_size = processed_inputs["input_ids"].shape[0]
                seq_len = vlm_output.image_hidden_states.shape[0] // batch_size
                image_features = vlm_output.image_hidden_states.view(batch_size, seq_len, -1)
            else:
                image_features = vlm_output.image_hidden_states

            last_hidden_state = vlm_output.hidden_states[-1]
            if self.vlm_backbone_feature_selection == "first_image":
                hidden_states = image_features[:, : self.num_img_tokens, :]
            elif self.vlm_backbone_feature_selection == "last_token":
                hidden_states = last_hidden_state[:, -1:, :]
            elif self.vlm_backbone_feature_selection == "all_generated":
                hidden_states = last_hidden_state
            elif self.vlm_backbone_feature_selection == "all":
                hidden_states = torch.cat((image_features, last_hidden_state), dim=1)
            elif self.vlm_backbone_feature_selection == "first_image_all":
                hidden_states = torch.cat(
                    (image_features[:, : self.num_img_tokens, :], last_hidden_state), dim=1
                )
            else:
                raise NotImplementedError(" not supported")
        else:
            raise NotImplementedError(f"{self.vlm_backbone_name} not implemented.")
        return hidden_states

    def get_action_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_states.shape
        hidden_states = hidden_states.transpose(0, 1)
        if self.action_decoder_name == "act_decoder":
            # Generate positional embeddings for the decoder
            decoder_pos_embeddings = self.decoder_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
            # Decode the action with positional embeddings and encoder output
            x = torch.zeros(
                (self.chunk_size, batch_size, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            action_logits = self.action_decoder(
                x=x, encoder_out=hidden_states, decoder_pos_embed=decoder_pos_embeddings
            )
            # Final action logits through the action head
            action_logits = action_logits.transpose(0, 1)
            action_logits = self.action_head(action_logits)
        elif self.action_decoder_name == "act":
            action_logits = self.action_decoder(
                x=x, encoder_in_tokens=hidden_states, encoder_in_pos_embed=None
            )
            # Final action logits through the action head
            action_logits = self.action_head(action_logits)
        else:
            raise NotImplementedError(f"{self.action_decoder_name} not supported.")

        return action_logits

    def forward(self, batch):
        """
        Forward pass to compute action logits.

        Args:
        batch: model input.

        Returns:
        action_logits: Tensor of predicted actions.
        """

        if "resnet" in self.vlm_backbone_name:
            images = torch.stack(batch["observation.images"], dim=0)
            hidden_states = self.vision_language_model(images)["feature_map"]
            hidden_states = einops.rearrange(hidden_states, "b c h w -> b (h w) c")
        else:
            prompt = self.apply_prompt_template(batch["prompt"], add_generation_prompt=True)
            
            batch_size = len(batch["observation.images"])
            with record_function("processor"):
                processed_inputs = self.processor(
                    text=[prompt] * batch_size,
                    images=list(batch["observation.images"]),
                    return_tensors="pt",
                    padding=True,
                    do_rescale=False,
                )
            # maybe pass the device to the processor before?
            with record_function("processed_inputs to cuda"):
                for k in processed_inputs.keys():
                    processed_inputs[k] = processed_inputs[k].to(device=batch["observation.state"].device)

            hidden_states = self.get_vlm_features(processed_inputs)
        if self.use_robot_state:
            robot_state = self.encoder_robot_state_input_proj(batch["observation.state"]).unsqueeze(1)
            hidden_states = torch.cat([hidden_states, robot_state], axis=1)
        
        if self.use_action_connector:
            hidden_states = self.action_connector(hidden_states)

        action_logits = self.get_action_logits(hidden_states)
        return action_logits
