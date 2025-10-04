#!/usr/bin/env python

from collections import deque

import numpy as np
import torch
from scipy.fft import idct
from torch import Tensor, nn

from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
from transformers import LogitsProcessorList

from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGE
from lerobot.policies.smolandfast.configuration_smolandfast import SMOLANDFASTConfig
from lerobot.policies.pretrained import PreTrainedPolicy

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class SMOLANDFASTPolicy(PreTrainedPolicy):
    """Wrapper class around PI0FAST tokenizer and SMOLANDFAST model to train and run inference within LeRobot."""

    config_class = SMOLANDFASTConfig
    name = "smolandfast"

    def __init__(
        self,
        config: SMOLANDFASTConfig,
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

        self.model = SMOLANDFAST(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        vision_model_params = self.model.vlm.model.vision_model.parameters()
        connector_params = self.model.vlm.model.connector.parameters()
        text_model_params = self.model.vlm.model.text_model.parameters()
        optim_groups = [{"params": vision_model_params, "lr": self.config.vision_model_optimizer_lr},
                        {"params": connector_params, "lr": self.config.connector_optimizer_lr},
                        {"params": text_model_params, "lr": self.config.text_model_optimizer_lr}]
        return optim_groups

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for SMOLANDFAST")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model.generate_actions(batch)

            actions = actions[:, : self.config.n_action_steps]

            original_action_dim = self.config.action_feature.shape[
                0
            ]  # self.config.max_action_dim  # self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        loss_dict = self.model.forward(batch)
        return loss_dict["loss"], loss_dict


class SMOLANDFAST(nn.Module):
    def __init__(self, config: SMOLANDFASTConfig):
        super().__init__()
        self.config = config

        self.torch_precision = PRECISION.get(config.precision, torch.float32)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.config.vlm_checkpoint, torch_dtype=self.torch_precision
        )
        self.processor = AutoProcessor.from_pretrained(self.config.vlm_checkpoint, use_fast=True)

        if config.scale_factor != 4:
            # if config factor is not 4 we need to recreate a linear layer in connector
            siglip_seq_len = (
                1024  # for 512x512 image with patch size 16 sequence len is 1024
            )
            self.processor.image_seq_len = int(
                siglip_seq_len / (config.scale_factor**2)
            )
            self.vlm.scale_factor = config.scale_factor
            self.vlm.model.connector.scale_factor = config.scale_factor
            input_size = self.vlm.config.vision_config.hidden_size * (
                config.scale_factor**2
            )
            output_size = self.vlm.config.text_config.hidden_size
            self.vlm.model.connector.modality_projection = nn.Linear(
                input_size, output_size, bias=False, dtype=self.torch_precision
            )

        fast_tokenizer_path = "physical-intelligence/fast"
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True
        )
        self.fast_skip_tokens = self.config.fast_skip_tokens
        self.max_input_seq_len = self.config.max_input_seq_len
        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[0]

        if config.freeze_vision_encoder:
            for param in self.vlm.model.vision_model.parameters():
                param.requires_grad = False

        if config.freeze_connector and config.scale_factor != 4:
            raise ValueError(
                "If scale factor is equal 4(default value) connector should be unfeezed"
            )

        if config.freeze_connector:
            for param in self.vlm.model.connector.parameters():
                param.requires_grad = False

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id

    def create_obs_prefix_tokens(self, states, images, lang_text):
        device = states.device

        # Precompute bin edges on GPU
        bins = torch.linspace(-1, 1, self.config.n_state_bins + 1, device=device)[:-1]

        # Discretize directly on GPU
        discretized_states = torch.bucketize(states, bins) - 1  # shape: [B, state_dim]

        # Move the batched results to CPU only once for string formatting
        disc_states_cpu = discretized_states.detach().cpu().numpy()

        # Build strings in batch
        prefix_texts = []
        for txt, disc_st in zip(lang_text, disc_states_cpu):
            cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(map(str, disc_st.tolist()))
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"Task: {cleaned}, State: {state_str}, Action: ",
                        },
                    ],
                }
            ]
            prefix_texts.append(message)

        prompts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True)
            for m in prefix_texts
        ]
        images = list(torch.unbind(images, dim=0))
        images = [[img] for img in images]

        prefix_out = self.processor(
            images=images,
            text=prompts,
            do_resize=self.config.do_image_spliting,
            do_rescale=False,
        )
        prefix_out["pixel_values"] = torch.tensor(
            np.array(prefix_out["pixel_values"]), dtype=torch.float32, device=device
        )
        prefix_out["pixel_attention_mask"] = torch.tensor(
            np.array(prefix_out["pixel_attention_mask"]),
            dtype=torch.long,
            device=device,
        )
        return prefix_out

    def _act_tokens_to_llm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        llm_tokens = (
            self.processor.tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        )
        return llm_tokens

    def _llm_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        fast_tokens = (
            self.processor.tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens
        )
        return fast_tokens

    def create_input_tokens(self, states, images, lang_text, actions=None):
        device = states.device

        prefix_out = self.create_obs_prefix_tokens(
            states=states, images=images, lang_text=lang_text
        )
        obs_ids = prefix_out["input_ids"]

        if actions is not None:
            fast_action_tokens = self.fast_tokenizer(actions.detach().cpu())

            llm_action_tokens = []
            for seq in fast_action_tokens:
                llm_seq = []
                for token in seq:
                    llm_seq.append(self._act_tokens_to_llm_tokens(token))
                llm_action_tokens.append(llm_seq)

            prefix_tokens = []
            loss_mask = []
            for obs_seq, act_seq in zip(obs_ids, llm_action_tokens):
                prefix_tokens.append(obs_seq + act_seq + [self.eos_token_id])
                loss_mask.append([0] * len(obs_seq) + [1] * len(act_seq) + [1])
            prefix_mask = [[1] * len(tokens) for tokens in prefix_tokens]

            max_len = max([len(seq) for seq in prefix_tokens])

            prefix_pad = []
            prefix_mask_pad = []
            loss_mask_pad = []

            # right padding for training
            for seq, seq_mask, seq_loss in zip(prefix_tokens, prefix_mask, loss_mask):
                seq_len = len(seq)
                prefix_pad.append(seq + [self.pad_token_id] * (max_len - seq_len))
                prefix_mask_pad.append(seq_mask + [0] * (max_len - seq_len))
                loss_mask_pad.append(seq_loss + [0] * (max_len - seq_len))

            prefix_tokens = torch.tensor(prefix_pad, dtype=torch.long).to(device)
            prefix_pad_mask = torch.tensor(prefix_mask_pad, dtype=torch.long).to(device)
            loss_mask = torch.tensor(loss_mask_pad, dtype=torch.long).to(device)

            return {
                "input_ids": prefix_tokens,
                "pad_masks": prefix_pad_mask,
                "pixel_values": prefix_out["pixel_values"],
                "pixel_attention_mask": prefix_out["pixel_attention_mask"],
                "loss_mask": loss_mask,
            }
        else:

            max_len = max([len(seq) for seq in obs_ids])
            prefix_tokens = []
            prefix_pad_mask = []

            # left padding for generation
            for seq in obs_ids:
                seq_len = len(seq)
                prefix_tokens.append([self.pad_token_id] * (max_len - seq_len) + seq)
                prefix_pad_mask.append([0] * (max_len - seq_len) + [1] * seq_len)

            prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long).to(device)
            prefix_pad_mask = torch.tensor(prefix_pad_mask, dtype=torch.long).to(device)

            return {
                "input_ids": prefix_tokens,
                "pad_masks": prefix_pad_mask,
                "pixel_values": prefix_out["pixel_values"],
                "pixel_attention_mask": prefix_out["pixel_attention_mask"],
                "loss_mask": None,
            }

    def forward(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        padded_outs = self.create_input_tokens(
            states=batch[OBS_STATE],
            images=batch[OBS_IMAGE],
            lang_text=batch["task"] if "task" in batch else "",
            actions=batch[ACTION],
        )

        outputs = self.vlm.forward(
            input_ids=padded_outs["input_ids"],
            attention_mask=padded_outs["pad_masks"],
            pixel_values=padded_outs["pixel_values"],
            pixel_attention_mask=padded_outs["pixel_attention_mask"],
            use_cache=self.config.use_cache,
        )

        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Shift left for next-step prediction
        logits = logits[:, :-1, :]
        targets = padded_outs["input_ids"][:, 1:].to(device)  # Shift targets
        loss_mask = padded_outs["loss_mask"][:, 1:].to(device)  # Ensure correct shape

        # Compute per-token loss
        token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        # Apply loss mask
        token_loss = token_loss * loss_mask.reshape(-1)

        # Compute final loss
        loss = token_loss.sum() / torch.clamp(loss_mask.sum(), min=1)

        # Return loss dictionary
        loss_dict = {"ce_loss": loss.item(), "loss": loss}
        return loss_dict

    def decode_actions_with_fast(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
        relaxed_decoding: bool = True,
    ) -> np.array:
        """
        Adapt original decoding in FAST to always return actions instead of zeros.
        """
        self.time_horizon = (
            time_horizon
            or self.fast_tokenizer.time_horizon
            or self.fast_tokenizer.called_time_horizon
        )
        self.action_dim = (
            action_dim
            or self.fast_tokenizer.action_dim
            or self.fast_tokenizer.called_action_dim
        )

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert (
            self.time_horizon is not None and self.action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = (
                    np.array(list(map(ord, decoded_tokens)))
                    + self.fast_tokenizer.min_token
                )
                if relaxed_decoding:
                    # Expected sequence length
                    expected_seq_len = self.time_horizon * self.action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]
                    # Apply truncation if too long
                    if diff < 0:
                        # Truncate on the right
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]
                    # Apply padding if too short
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff,
                            (0, diff),
                            mode="constant",
                            constant_values=0,
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert decoded_dct_coeff.shape == (
                    self.time_horizon,
                    self.action_dim,
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({self.time_horizon}, {self.action_dim})"
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(
                idct(
                    decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"
                )
            )
        return np.stack(decoded_actions)

    def generate_actions(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        padded_outs = self.create_input_tokens(
            states=batch[OBS_STATE],
            images=batch[OBS_IMAGE],
            lang_text=batch["task"] if "task" in batch else "",
            actions=None,
        )

        input_len = padded_outs["input_ids"].shape[1]

        def make_fast_band_processor(low, high, special_tokens):
            def processor(input_ids, scores):
                # Everything outside [low, high] and eos_id → -inf
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask[:, low : high + 1] = False
                for token in special_tokens:
                    mask[:, token] = False
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
                return scores

            return processor

        # Example usage in generate_actions
        fast_vocab_size = self.fast_tokenizer.bpe_tokenizer.vocab_size
        high = self.processor.tokenizer.vocab_size - 1 - self.fast_skip_tokens
        low = high - (fast_vocab_size - 1)

        processors = LogitsProcessorList(
            [
                make_fast_band_processor(
                    low, high, [self.eos_token_id, self.pad_token_id]
                )
            ]
        )

        output_tokens = self.vlm.generate(
            input_ids=padded_outs["input_ids"],
            attention_mask=padded_outs["pad_masks"],
            pixel_values=padded_outs["pixel_values"],
            pixel_attention_mask=padded_outs["pixel_attention_mask"],
            use_cache=self.config.use_cache,
            max_new_tokens=self.config.max_decoding_steps,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            logits_processor=processors,
        )
        gemma_action_tokens = output_tokens[:, input_len:]

        fast_eos_token = self._llm_tokens_to_act_tokens(self.eos_token_id)
        fast_pad_token = self._llm_tokens_to_act_tokens(self.pad_token_id)

        fast_action_tokens = self._llm_tokens_to_act_tokens(
            gemma_action_tokens
        ).tolist()

        # remove fast pad tokens and eos token
        for seq in fast_action_tokens:
            while seq and (seq[-1] == fast_eos_token or seq[-1] == fast_pad_token):
                seq.pop()

        decoded_actions = torch.tensor(
            [
                self.decode_actions_with_fast(
                    [tok],
                    time_horizon=self.action_horizon,
                    action_dim=self.action_dim,
                    relaxed_decoding=self.config.relaxed_action_decoding,
                ).squeeze(0)
                for tok in fast_action_tokens
            ],
            dtype=torch.float32,
            device=device,
        )
        return decoded_actions
