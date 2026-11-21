# Copyright 2026 Alibaba DAMO Academy and The HuggingFace Inc. team. All rights reserved.
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

# processing_rynn_value_lang.py


import torch
from einops import rearrange
from PIL.Image import Image
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLVideoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from .conversations import (
    ConversationBuilder,
    InterleavedHistoryConversationBuilder,
    build_conversation_builder,
)

DEFAULT_CONVERSATION_STYLE = InterleavedHistoryConversationBuilder.name


class RynnValueLangProcessor(Qwen3VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        use_meta=False,
        conversation_style: str = DEFAULT_CONVERSATION_STYLE,
        value_token_repeat: int = 1,
        relative_value_token_repeat: int = 1,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        # Whether to inject robot type / camera position meta info into prompts.
        # Decided at processor construction and persisted in processor_config.json;
        # not threaded through individual process_* calls.
        self.use_meta = use_meta
        # Selects the ConversationBuilder that owns the prompt layout. Persisted
        # in processor_config.json so train and inference build identical prompts
        # without the caller having to re-specify.
        self.conversation_style = conversation_style
        # Number of consecutive <value> tokens emitted per prediction slot.
        # Persisted so the model can pool the R tokens back to one prediction.
        if value_token_repeat < 1:
            raise ValueError(f"value_token_repeat must be >= 1, got {value_token_repeat}")
        self.value_token_repeat = int(value_token_repeat)
        # Same for <relative_value>: R copies per slot, pooled by the model.
        if relative_value_token_repeat < 1:
            raise ValueError(f"relative_value_token_repeat must be >= 1, got {relative_value_token_repeat}")
        self.relative_value_token_repeat = int(relative_value_token_repeat)
        # Build eagerly so callers can rely on `self.conversation_builder` from
        # __init__ onwards. After mutating tokens or toggling use_meta /
        # conversation_style, callers must invoke
        # ``refresh_conversation_builder()`` to pick up the change — see
        # ``add_special_tokens_and_resize``.
        self.conversation_builder: ConversationBuilder = self._build_conversation_builder()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load checkpoint assets explicitly without executing Hub model code.

        Released RynnValue processor metadata predates Transformers 5 and does
        not describe its video processor. Constructing the known Qwen
        components directly avoids both that compatibility issue and
        ``trust_remote_code``.
        """
        processor_dict, _ = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        load_keys = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        }
        load_kwargs = {key: value for key, value in kwargs.items() if key in load_keys}
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
            fix_mistral_regex=False,
            **load_kwargs,
        )
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(),
            chat_template=getattr(tokenizer, "chat_template", None),
            use_meta=kwargs.get("use_meta", processor_dict.get("use_meta", False)),
            conversation_style=kwargs.get(
                "conversation_style",
                processor_dict.get("conversation_style", DEFAULT_CONVERSATION_STYLE),
            ),
            value_token_repeat=kwargs.get("value_token_repeat", processor_dict.get("value_token_repeat", 1)),
            relative_value_token_repeat=kwargs.get(
                "relative_value_token_repeat",
                processor_dict.get("relative_value_token_repeat", 1),
            ),
        )

    @classmethod
    def from_qwen3vl(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Build a RynnValueLangProcessor from a Qwen3VL checkpoint.
        """
        return cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @property
    def value_token(self):
        return "<value>"

    @property
    def value_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.value_token)

    @property
    def relative_value_token(self):
        return "<relative_value>" if "<relative_value>" in self.tokenizer.get_vocab() else None

    @property
    def relative_value_token_id(self):
        token = self.relative_value_token
        return None if token is None else self.tokenizer.convert_tokens_to_ids(token)

    def __call__(
        self,
        images: ImageInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        return super().__call__(
            images=images,
            text=text,
            **kwargs,
            **{
                "return_tensors": "pt",
            },
        )

    def _build_conversation_builder(self) -> ConversationBuilder:
        """Snapshot the processor's current state into a ConversationBuilder."""
        return build_conversation_builder(
            self.conversation_style,
            value_token=self.value_token,
            relative_value_token=self.relative_value_token,
            use_meta=self.use_meta,
            value_token_repeat=self.value_token_repeat,
            relative_value_token_repeat=self.relative_value_token_repeat,
        )

    def refresh_conversation_builder(self) -> ConversationBuilder:
        """Rebuild ``self.conversation_builder`` from the current processor state.

        Callers should invoke this whenever they mutate any input the builder
        depends on (e.g. ``use_meta``, ``conversation_style``, or any of the
        special tokens on the tokenizer).
        """
        self.conversation_builder = self._build_conversation_builder()
        return self.conversation_builder

    @staticmethod
    def _frame_target_values(
        goal_timestamp: torch.Tensor,
        frame_timestamps: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-frame targets as [1, N]."""
        ts = frame_timestamps.to(device).reshape(-1)
        goal_ts = goal_timestamp.to(device).reshape(1)
        return (goal_ts - ts).float().unsqueeze(0)

    @staticmethod
    def _anchor_target_value(
        goal_timestamp: torch.Tensor,
        frame_timestamps: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the single anchor (last-frame) target as [1, 1]."""
        ts = frame_timestamps.to(device).reshape(-1)
        goal_ts = goal_timestamp.to(device).reshape(1)
        return (goal_ts - ts[-1:]).float().unsqueeze(0)

    def _progress_targets(
        self,
        goal_timestamp: torch.Tensor,
        frame_timestamps: torch.Tensor,
        device: torch.device,
        value_count: int,
    ) -> torch.Tensor:
        """Targets aligned with the builder's <value> token count.

        ``value_count == 1`` -> anchor (last-frame) remaining time, [1, 1].
        Otherwise per-frame remaining times, [1, value_count]; ``value_count``
        must equal the number of input frames so the targets align with the
        interleaved tokens. Frames past the goal saturate at 0 — the progress
        head represents non-negative time-to-done.
        """
        if value_count == 1:
            target = self._anchor_target_value(goal_timestamp, frame_timestamps, device)
        else:
            target = self._frame_target_values(goal_timestamp, frame_timestamps, device)
            if target.shape[1] != value_count:
                raise ValueError(
                    f"Conversation builder expects {value_count} <value> targets but "
                    f"received {target.shape[1]} frame timestamps."
                )
        return target.clamp_min(0)

    def process_history(
        self,
        instruction: str,
        description: str | None,
        images: list[Image],
        frame_timestamps: torch.Tensor,
        goal_timestamp: torch.Tensor,
        success: bool | None = None,
        fusion: bool | None = None,
        robot_description: str | None = None,
        camera_description: str | None = None,
    ) -> BatchFeature:
        """Process a history sample.

        Layout (``interleaved_history``): top-level instruction, frames with
        ``<relative_value>`` between them, optional description sentence
        (input only, no LM supervision), and per-frame ``<value>`` tokens.

        Relative-value targets are derived from ``frame_timestamps`` as
        signed consecutive deltas of length ``num_frames - 1``.

        Token positions for the prediction heads are inferred inside the
        model from ``input_ids`` + registered token IDs, so no
        ``*_logits_to_keep`` tensors are emitted here.
        """
        builder = self.conversation_builder
        num_frames = len(images)
        value_count = builder.progress_value_count(num_frames)

        fusion_flag = 0 if fusion is None else int(bool(fusion))
        match_answer = "No" if fusion_flag else "Yes"
        success_answer = None if success is None else "Yes" if success else "No"

        content = builder.build_progress(
            instruction=instruction,
            description=description,
            num_frames=num_frames,
            robot_description=robot_description,
            camera_description=camera_description,
            match_answer=match_answer,
            success_answer=success_answer,
        )

        conversation = [{"role": "user", "content": content}]
        text = self.apply_chat_template(conversation)

        mini_outputs = self.__call__(text=text, images=images)
        input_ids = mini_outputs["input_ids"]

        target_values = self._progress_targets(
            goal_timestamp,
            frame_timestamps,
            device=input_ids.device,
            value_count=value_count,
        )
        mini_outputs["value"] = target_values

        mini_outputs["value_fusion_mask"] = torch.full_like(
            target_values, fill_value=fusion_flag, dtype=torch.long
        )

        ts = frame_timestamps.reshape(-1).to(input_ids.device).float()
        mini_outputs["relative_value"] = (ts[1:] - ts[:-1]).unsqueeze(0)

        # Language loss labels: supervise the "Analysis:" span with causal LM
        # loss. Always emitted (all -100 if marker not found) so batches with
        # mixed samples concat cleanly.
        mini_outputs["labels"] = self._build_history_labels(input_ids)

        return mini_outputs

    def _build_history_labels(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Create causal LM labels for the "Analysis:" span in the history layout.

        Everything after "Analysis:" is supervised with causal LM loss.
        Since "Analysis:" is emitted as a standalone text block, its token
        sequence is deterministic — we match it directly in input_ids.
        Raises ValueError if the marker isn't found (indicates a builder bug).
        """
        batch_size, seq_len = input_ids.shape
        marker_ids = self.tokenizer.encode("Analysis: \n", add_special_tokens=False)
        marker_len = len(marker_ids)
        marker_t = torch.tensor(marker_ids, device=input_ids.device)

        labels = torch.full_like(input_ids, -100)

        for b in range(batch_size):
            seq = input_ids[b]
            found = False
            for i in range(seq_len - marker_len, -1, -1):
                if torch.equal(seq[i : i + marker_len], marker_t):
                    content_start = i + marker_len
                    if content_start < seq_len:
                        labels[b, content_start:] = seq[content_start:]
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"'Analysis:' marker not found in sample {b}. "
                    "The conversation builder must emit an 'Analysis:' block."
                )

        return labels

    def process_episode(
        self,
        instruction: str,
        images: list[Image],
        robot_description: str | None = None,
        camera_description: str | None = None,
    ) -> BatchFeature:
        """Process one episode in progress mode (inference).

        All images passed in are used directly as prediction frames.
        The output is shaped as a single sample: ``input_ids``/``attention_mask``
        are ``(1, L)`` and ``pixel_values``/``image_grid_thw`` are ``(1, k, D)``
        with ``k`` equal to the number of input images.
        """
        if len(images) == 0:
            raise ValueError("process_episode requires at least one prediction image.")

        k = len(images)
        builder = self.conversation_builder

        content = builder.build_progress(
            instruction=instruction,
            description=None,
            num_frames=k,
            robot_description=robot_description,
            camera_description=camera_description,
            is_inference=True,
        )
        conv = [{"role": "user", "content": content}]

        text = self.apply_chat_template(conv)
        merged = self.__call__(text=text, images=images)

        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        input_ids = merged["input_ids"]
        eos_positions = (input_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            input_ids = input_ids[:, : eos_positions[-1]]
        merged["input_ids"] = input_ids
        merged["attention_mask"] = merged["attention_mask"][:, : input_ids.shape[-1]]
        if "mm_token_type_ids" in merged:
            merged["mm_token_type_ids"] = merged["mm_token_type_ids"][:, : input_ids.shape[-1]]

        merged["pixel_values"] = rearrange(merged["pixel_values"], "(b t) d -> b t d", b=1)
        merged["image_grid_thw"] = rearrange(merged["image_grid_thw"], "(b t) d -> b t d", b=1)
        return merged
