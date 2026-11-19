# Copyright 2026 Ahad Jawaid. All rights reserved.
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

"""NAT VLA: a vision-language model writing NAT action tokens, decoded into actions as they are generated."""

import json
import threading
from functools import partial
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, load_model
from torch import Tensor
from torch.nn import functional

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_nat_vla import NATVLAConfig, downloaded
from .nat_tokenizer import NATTokenizer, StreamingDecoder
from .processor_nat_vla import IMAGE_INPUTS, Vocabulary, padded_token_ids, vlm_config_of

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, Lfm2VlForConditionalGeneration
    from transformers.generation.streamers import BaseStreamer
else:
    AutoConfig = Lfm2VlForConditionalGeneration = None
    BaseStreamer = object


class NATVLAPolicy(PreTrainedPolicy):
    config_class = NATVLAConfig
    name = "nat_vla"
    _fsdp_wrap_modules = ["Lfm2DecoderLayer", "Siglip2EncoderLayer"]

    def __init__(self, config: NATVLAConfig, **kwargs):
        require_package("transformers", extra="nat_vla")
        super().__init__(config)
        config.validate_features()
        if config.action_feature is None:
            raise ValueError("NAT VLA needs an 'action' output feature.")
        self.config, self.action_dim = config, config.action_feature.shape[0]

        tokenizer_folder = None
        if config.action_tokenizer_config is None:
            tokenizer_folder = downloaded(config.action_tokenizer, ["config.json", "model.safetensors"])
            config.action_tokenizer_config = json.loads((tokenizer_folder / "config.json").read_text())

        self.action_tokenizer = NATTokenizer(config.action_tokenizer_config).float().requires_grad_(False)
        if tokenizer_folder is not None:
            self.action_tokenizer.load_state_dict(load_file(tokenizer_folder / "model.safetensors"))
        self.action_tokenizer.register_state_dict_post_hook(lstm_weights_copy_hook)

        tokenizer = self.action_tokenizer
        if config.chunk_size != tokenizer.time_horizon:
            raise ValueError(f"chunk_size must be the NAT tokenizer's time horizon, {tokenizer.time_horizon}")
        if config.rtc_steps and config.rtc_steps not in tokenizer.prefix_token_counts:
            raise ValueError(
                f"rtc_steps must be one of {sorted(tokenizer.prefix_token_counts)} (whole latents)"
            )
        if config.rtc_steps and self.action_dim > tokenizer.group_width:
            raise ValueError(f"rtc_steps needs actions of at most {tokenizer.group_width} columns")

        starts_from_pretrained = config.vlm_config is None and config.pretrained_path is None
        if config.vlm_config is None:
            config.vlm_config = vlm_config_of(downloaded(config.vlm, ["vlm/*"]) / "vlm")

        vlm_config = AutoConfig.for_model(**config.vlm_config)
        text_config = vlm_config.text_config
        self.vocabulary = Vocabulary(text_config.vocab_size, config.state_bins, tokenizer.vocab_size)
        self.eos_token_id, self.pad_token_id = text_config.eos_token_id, text_config.pad_token_id
        text_config.vocab_size = self.vocabulary.size
        self.vlm = Lfm2VlForConditionalGeneration._from_config(
            vlm_config, attn_implementation="sdpa", dtype=config.dtype
        )
        self.vlm.get_input_embeddings().weight.register_hook(
            partial(added_rows_only_gradient_hook, first_added_row=self.vocabulary.setup_start)
        )

        if starts_from_pretrained:
            load_model(self, downloaded(config.vlm, ["model.safetensors"]) / "model.safetensors")
        self.reset()

    @property
    def device(self) -> torch.device:
        return self.vlm.device

    def reset(self):
        for stream in (getattr(self, "_stream", None), getattr(self, "_next_stream", None)):
            if stream is not None:
                stream.thread.join()

        self._stream = self._next_stream = None
        self._chunk_step = self._chunk_end_step = 0

    def supports_rtc(self) -> bool:
        """Real-time chunking through the tokens: the executing actions start the next chunk as a token prefix."""
        return self.action_dim <= self.action_tokenizer.group_width

    def get_optim_params(self) -> list[dict]:
        vision_modules = (self.vlm.model.vision_tower, self.vlm.model.multi_modal_projector)
        vision = {id(parameter) for module in vision_modules for parameter in module.parameters()}
        trained = [parameter for parameter in self.parameters() if parameter.requires_grad]

        return [
            {"params": [parameter for parameter in trained if id(parameter) not in vision]},
            {
                "params": [parameter for parameter in trained if id(parameter) in vision],
                "lr": self.config.optimizer_vision_lr,
            },
        ]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        prompts, image_inputs = self.prompts_and_image_inputs(batch)
        vocabulary, actions = self.vocabulary, batch[ACTION]
        action_tokens = self.action_tokenizer.encode(actions) + vocabulary.first_action_token
        padded_steps = batch.get(f"{ACTION}_is_pad", torch.zeros_like(actions[..., 0], dtype=torch.bool))
        tokens_with_actions = self.action_tokenizer.tokens_with_actions(~padded_steps, self.action_dim)

        outputs = [
            [vocabulary.action_start, *tokens, vocabulary.action_end, self.eos_token_id]
            for tokens in action_tokens.tolist()
        ]
        output_masks = [[True, *mask, True, True] for mask in tokens_with_actions.tolist()]
        input_ids, attention_mask = padded_token_ids(
            [prompt + output for prompt, output in zip(prompts, outputs, strict=True)],
            "right",
            self.pad_token_id,
        )

        hidden_states = self.vlm.model(
            input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), **image_inputs
        ).last_hidden_state

        samples, positions, weights = output_token_positions(prompts, outputs, output_masks)
        logits = self.vlm.lm_head(hidden_states[samples, positions]).float()
        weights = weights.to(logits.device)
        cross_entropy = functional.cross_entropy(
            logits, input_ids[samples, positions + 1].to(logits.device), reduction="none"
        )

        z_loss = self.config.z_loss_scale * logits.logsumexp(-1).square()
        cross_entropy, z_loss = ((weights * value).sum() / weights.sum() for value in (cross_entropy, z_loss))
        return cross_entropy + z_loss, {"cross_entropy": cross_entropy.item(), "z_loss": z_loss.item()}

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        steps: int | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        """Actions for the first `steps` steps (the whole chunk by default), rounded up to whole NAT latents. With
        real-time chunking, the chunk starts with the tokens of the previous chunk's actions that run during
        inference_delay steps, rounded up to whole latents."""
        tokenizer = self.action_tokenizer
        steps = tokenizer.token_count(steps or self.config.chunk_size, self.action_dim)[1]
        executing_actions = None
        if inference_delay and prev_chunk_left_over is not None and self.supports_rtc():
            left_over = prev_chunk_left_over.reshape(-1, *prev_chunk_left_over.shape[-2:])
            covering_steps = [count for count in tokenizer.prefix_token_counts if count >= inference_delay]
            if covering_steps and covering_steps[0] < steps and covering_steps[0] <= left_over.shape[1]:
                executing_actions = left_over[:, : covering_steps[0], : self.action_dim].to(self.device)

        stream = self.stream_action_chunk(batch, steps, executing_actions)
        stream.thread.join()
        if stream.error is not None:
            raise stream.error
        return tokenizer.decode(stream.decoder.tokens, self.action_dim)[:, :steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """The next action of the streamed chunk; a new chunk starts every n_action_steps actions. With
        rtc_steps, the next chunk is generated while the current chunk's last rtc_steps actions run, starting
        with their tokens."""
        steps_per_chunk, rtc_steps = self.config.n_action_steps, self.config.rtc_steps
        if self._chunk_step == self._chunk_end_step:
            if self._next_stream is None:
                self.reset()
                self._stream = self.stream_action_chunk(batch, steps_per_chunk)
            else:
                self._stream.thread.join()
                self._stream, self._next_stream, self._chunk_step = self._next_stream, None, rtc_steps
            self._chunk_end_step = self._chunk_step + steps_per_chunk

        if rtc_steps and self._chunk_step == self._chunk_end_step - rtc_steps:
            executing_steps = range(self._chunk_step, self._chunk_end_step)
            executing_actions = torch.stack([self._stream.action(step) for step in executing_steps], 1)
            self._stream.thread.join()
            self._next_stream = self.stream_action_chunk(
                batch, rtc_steps + steps_per_chunk, executing_actions
            )

        self._chunk_step += 1
        return self._stream.action(self._chunk_step - 1)

    def stream_action_chunk(
        self, batch: dict[str, Tensor], steps: int, executing_actions: Tensor | None = None
    ) -> "ActionStreamHook":
        """Generates the chunk's first `steps` steps in a background thread, handing over each latent's
        actions once decoded. With executing_actions, the chunk starts with their tokens and their actions are
        not handed over again."""
        tokenizer, vocabulary = self.action_tokenizer, self.vocabulary
        token_count = tokenizer.token_count(steps, self.action_dim)[0]

        first_tokens = None
        prefixes: list[list[int]] = [[] for _ in batch["input_ids"]]
        if executing_actions is not None:
            first_tokens = tokenizer.tokens_starting_with(executing_actions)
            prefixes = (first_tokens + vocabulary.first_action_token).tolist()

        prompts, image_inputs = self.prompts_and_image_inputs(batch)
        input_ids, attention_mask = padded_token_ids(
            [
                prompt + [vocabulary.action_start, *prefix]
                for prompt, prefix in zip(prompts, prefixes, strict=True)
            ],
            "left",
            self.pad_token_id,
        )

        stream = ActionStreamHook(
            tokenizer.stream(self.action_dim, first_tokens), vocabulary.first_action_token
        )
        generate = partial(
            self.vlm.generate,
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            **image_inputs,
            max_new_tokens=token_count - len(prefixes[0]),
            do_sample=False,
            pad_token_id=self.pad_token_id,
            logits_processor=[
                partial(action_tokens_only_hook, first_action_token=vocabulary.first_action_token)
            ],
        )

        stream.start(generate, self.device)
        return stream

    def prompts_and_image_inputs(self, batch: dict[str, Tensor]) -> tuple[list[list[int]], dict[str, Tensor]]:
        """The prompt token ids and image inputs the preprocessor built."""
        prompts = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"], strict=True)
        ]
        image_inputs = {name: batch[name].to(self.device) for name in IMAGE_INPUTS}
        image_inputs["pixel_values"] = image_inputs["pixel_values"].to(self.vlm.dtype)
        return prompts, image_inputs


class ActionStreamHook(BaseStreamer):
    """Generation hook: generate() hands it the prompt, then each new token. It feeds the tokens to the
    tokenizer's streaming decoder and keeps the decoded actions for action(step), which waits until that step
    is decoded."""

    def __init__(self, decoder: StreamingDecoder, first_action_token: int):
        self.decoder, self.first_action_token = decoder, first_action_token
        self.first_step, self.prompt_seen = decoder.decoded_steps, False
        self.actions: list[Tensor] = []
        self.error, self.finished, self.arrived = None, False, threading.Condition()

    def start(self, generate, device: torch.device):
        def generated():
            try:
                generate(streamer=self)
            except BaseException as error:
                self.error = error
            with self.arrived:
                self.finished = True
                self.arrived.notify_all()

        self.device, self.thread = device, threading.Thread(target=generated)
        self.thread.start()

    def put(self, value: Tensor):
        if not self.prompt_seen:
            self.prompt_seen = True
            return

        actions = self.decoder.push(value.to(self.device) - self.first_action_token)
        if actions is not None:
            with self.arrived:
                self.actions.extend(actions.unbind(1))
                self.arrived.notify_all()

    def end(self):
        pass

    def action(self, step: int) -> Tensor:
        """The action of the chunk's step `step`, once decoded."""
        with self.arrived:
            self.arrived.wait_for(lambda: self.finished or step < self.first_step + len(self.actions))

        if step >= self.first_step + len(self.actions):
            raise self.error or RuntimeError(f"generation ended before step {step}")
        return self.actions[step - self.first_step]


def lstm_weights_copy_hook(module, state_dict, prefix, local_metadata):
    """State-dict hook: on the GPU the LSTM's weights are views of one flat buffer, which safetensors refuses
    to save."""
    for name in [name for name in state_dict if name.startswith(f"{prefix}encoder.lstm.")]:
        state_dict[name] = state_dict[name].clone()


def added_rows_only_gradient_hook(gradient: Tensor, first_added_row: int) -> Tensor:
    return torch.cat([torch.zeros_like(gradient[:first_added_row]), gradient[first_added_row:]])


def output_token_positions(
    prompts: list[list[int]], outputs: list[list[int]], output_masks: list[list[bool]]
) -> tuple[Tensor, Tensor, Tensor]:
    """Each output token's sample, the position predicting it, and its loss weight: one over the square root of
    the sample's trained tokens, and zero for action tokens that describe only padded steps."""
    samples: list[int] = []
    positions: list[int] = []
    weights: list[float] = []
    for sample, (prompt, output, mask) in enumerate(zip(prompts, outputs, output_masks, strict=True)):
        samples += [sample] * len(output)
        positions += range(len(prompt) - 1, len(prompt) + len(output) - 1)
        weights += [kept / sum(mask) ** 0.5 for kept in mask]

    return torch.tensor(samples), torch.tensor(positions), torch.tensor(weights)


def action_tokens_only_hook(input_ids: Tensor, scores: Tensor, first_action_token: int) -> Tensor:
    """Generation hook: before each token is picked, sets the score of every non-action token to -inf."""
    scores[:, :first_action_token] = -torch.inf
    return scores
