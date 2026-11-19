# Copyright (c) 2026 Ahad Jawaid
"""NAT VLA: a vision-language model writing NAT action tokens, decoded into actions as they are generated."""

import json
import threading
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import functional

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_nat_vla import NATVLAConfig
from .nat_tokenizer import NATTokenizer, StreamingDecoder

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoProcessor, Lfm2VlForConditionalGeneration
    from transformers.generation.streamers import BaseStreamer
else:
    AutoConfig = AutoProcessor = Lfm2VlForConditionalGeneration = None
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

        tokenizer_folder = None if config.action_tokenizer_config else Path(config.action_tokenizer)
        if tokenizer_folder is not None:
            if not tokenizer_folder.is_dir():
                files = ["config.json", "model.safetensors"]
                tokenizer_folder = Path(snapshot_download(str(tokenizer_folder), allow_patterns=files))
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

        vlm_config = AutoConfig.from_pretrained(config.vlm, subfolder="vlm")
        self.vocabulary = Vocabulary(
            vlm_config.text_config.vocab_size, config.state_bins, tokenizer.vocab_size
        )
        vlm_config.text_config.vocab_size = self.vocabulary.size
        self.vlm = Lfm2VlForConditionalGeneration._from_config(
            vlm_config, attn_implementation="sdpa", dtype=config.dtype
        )
        self.vlm.get_input_embeddings().weight.register_hook(
            partial(added_rows_only_gradient_hook, first_added_row=self.vocabulary.setup_start)
        )

        self.processor = AutoProcessor.from_pretrained(config.vlm, subfolder="vlm")
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
        vocabulary = self.vocabulary
        action_tokens = self.action_tokenizer.encode(batch[ACTION]) + vocabulary.first_action_token
        outputs = [
            [vocabulary.action_start, *tokens, vocabulary.action_end, self.processor.tokenizer.eos_token_id]
            for tokens in action_tokens.tolist()
        ]
        input_ids, attention_mask = padded_token_ids(
            [prompt + output for prompt, output in zip(prompts, outputs, strict=True)], "right"
        )

        hidden_states = self.vlm.model(
            input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), **image_inputs
        ).last_hidden_state

        samples, positions, weights = output_token_positions(prompts, outputs)
        logits = self.vlm.lm_head(hidden_states[samples, positions]).float()
        weights = weights.to(logits.device)
        cross_entropy = functional.cross_entropy(
            logits, input_ids[samples, positions + 1].to(logits.device), reduction="none"
        )

        z_loss = self.config.z_loss_scale * logits.logsumexp(-1).square()
        cross_entropy, z_loss = ((weights * value).sum() / weights.sum() for value in (cross_entropy, z_loss))
        return cross_entropy + z_loss, {"cross_entropy": cross_entropy.item(), "z_loss": z_loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], steps: int | None = None) -> Tensor:
        """Actions for the first `steps` steps, rounded up to whole NAT latents."""
        steps = self.action_tokenizer.token_count(steps or self.config.chunk_size, self.action_dim)[1]
        stream = self.stream_action_chunk(batch, steps)
        actions = torch.stack([stream.action(step) for step in range(steps)], 1)

        stream.thread.join()
        return actions

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
        self.eval()
        tokenizer, vocabulary = self.action_tokenizer, self.vocabulary
        token_count = tokenizer.token_count(steps, self.action_dim)[0]

        first_tokens = None
        prefixes: list[list[int]] = [[] for _ in batch["task"]]
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
            pad_token_id=0,
            logits_processor=[
                partial(action_tokens_only_hook, first_action_token=vocabulary.first_action_token)
            ],
        )

        stream.start(generate, self.device)
        return stream

    def prompts_and_image_inputs(self, batch: dict[str, Tensor]) -> tuple[list[list[int]], dict[str, Tensor]]:
        cameras = [
            (batch[key] * 255).round().clamp(0, 255).to(torch.uint8) for key in self.config.image_features
        ]
        frames = [camera[sample] for sample in range(len(batch["task"])) for camera in cameras]
        processed_images = self.processor.image_processor(
            frames, return_row_col_info=True, return_tensors="pt"
        )
        image_prompts = [
            self.processor.replace_image_token(processed_images, index, use_image_special_tokens=True)
            for index in range(len(frames))
        ]

        states = batch.get(OBS_STATE, [None] * len(frames))
        prompts = [
            self.prompt_tokens(
                task, image_prompts[sample * len(cameras) : (sample + 1) * len(cameras)], states[sample]
            )
            for sample, task in enumerate(batch["task"])
        ]

        image_input_names = ("pixel_values", "pixel_attention_mask", "spatial_shapes")
        image_inputs = {name: processed_images[name].to(self.device) for name in image_input_names}
        image_inputs["pixel_values"] = image_inputs["pixel_values"].to(self.vlm.dtype)
        return prompts, image_inputs

    def prompt_tokens(self, task: str, image_prompts: list[str], state: Tensor | None) -> list[int]:
        vocabulary, tokenizer = self.vocabulary, self.processor.tokenizer
        images = "".join(f"Image {number}{prompt}" for number, prompt in enumerate(image_prompts, 1))

        prompt_parts: list[str | int] = [
            f"{tokenizer.bos_token}{images}<|im_start|>user\nThe task is to {task}. The setup is "
        ]
        if self.config.setup:
            prompt_parts += [vocabulary.setup_start, self.config.setup, vocabulary.setup_end]

        if state is not None:
            state_tokens = (
                state_bin_indices(state, self.config.state_bins) + vocabulary.first_state_bin
            ).tolist()
            prompt_parts += [
                ". The current state of the robot is ",
                vocabulary.state_start,
                *state_tokens,
                vocabulary.state_end,
            ]

        prompt_parts.append(". The expected control mode is ")
        if self.config.control_mode:
            prompt_parts += [vocabulary.control_start, self.config.control_mode, vocabulary.control_end]

        prompt_parts += [
            ". Given these, what action should the robot take to complete the task?"
            f"{tokenizer.eos_token}\n<|im_start|>assistant\n",
            vocabulary.action_output,
        ]

        encoded_parts = (
            [part] if isinstance(part, int) else tokenizer.encode(part, add_special_tokens=False)
            for part in prompt_parts
        )
        return [token for part in encoded_parts for token in part]


class Vocabulary:
    """Token ids of the VLM's text vocabulary followed by the added ones: prompt markers, state bins and NAT
    action tokens."""

    def __init__(self, text_vocabulary_size: int, state_bins: int, action_tokens: int):
        start = text_vocabulary_size
        self.setup_start, self.setup_end, self.control_start, self.control_end = range(start, start + 4)
        self.state_start, self.state_end, self.first_state_bin = range(start + 4, start + 7)

        self.action_output = self.first_state_bin + state_bins
        self.action_start, self.action_end, self.first_action_token = range(
            self.action_output + 1, self.action_output + 4
        )

        self.size = self.first_action_token + action_tokens


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
                with (
                    torch.no_grad(),
                    torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"),
                ):
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

        with torch.autocast(self.device.type, enabled=False):
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


def padded_token_ids(sequences: list[list[int]], side: str) -> tuple[Tensor, Tensor]:
    """Token id sequences padded with 0 on `side`, and their attention mask."""
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    positions = torch.arange(int(lengths.max()))
    mask = positions >= len(positions) - lengths[:, None] if side == "left" else positions < lengths[:, None]

    ids = torch.zeros(mask.shape, dtype=torch.long)
    ids[mask] = torch.tensor([token for sequence in sequences for token in sequence])
    return ids, mask.long()


def output_token_positions(
    prompts: list[list[int]], outputs: list[list[int]]
) -> tuple[Tensor, Tensor, Tensor]:
    """Each output token's sample, the position predicting it, and its loss weight."""
    samples: list[int] = []
    positions: list[int] = []
    weights: list[float] = []
    for sample, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
        samples += [sample] * len(output)
        positions += range(len(prompt) - 1, len(prompt) + len(output) - 1)
        weights += [len(output) ** -0.5] * len(output)

    return torch.tensor(samples), torch.tensor(positions), torch.tensor(weights)


def action_tokens_only_hook(input_ids: Tensor, scores: Tensor, first_action_token: int) -> Tensor:
    """Generation hook: before each token is picked, sets the score of every non-action token to -inf."""
    scores[:, :first_action_token] = -torch.inf
    return scores


def state_bin_indices(state: Tensor, bins: int) -> Tensor:
    """State in [-1, 1] as bin indices."""
    state = torch.nan_to_num(state.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    return torch.round((state + 1) / 2 * (bins - 1)).long()
