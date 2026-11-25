# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/sana_robot_ar.py. Import paths are rewritten to the LeRobot package;
# the streaming entries and the forward on the teacher-forcing sequence of one window; the packed row of several windows and the row modulation kernels are not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from lerobot.policies.sana_robot.sana_robot_core.block import PolicyBlock
from lerobot.policies.sana_robot.sana_robot_core.causal_cross_attention import (
    ChunkTextCrossAttention,
    chunk_text_spans,
)
from lerobot.policies.sana_robot.sana_robot_core.causal_gdn import CausalGatedDeltaNet
from lerobot.policies.sana_robot.sana_robot_core.causal_softmax import CausalSoftmaxAttention
from lerobot.policies.sana_robot.sana_robot_core.dit import SanaRobotBackbone
from lerobot.policies.sana_robot.sana_robot_core.multiview import (
    batch_view_shapes,
    batch_view_slot_ids,
    strip_to_view_tokens,
    view_to_strip_tokens,
)
from lerobot.policies.sana_robot.sana_robot_core.robot import ROBOT_DIM
from lerobot.policies.sana_robot.sana_robot_core.teacher_forcing import (
    latent_centre_offset,
    noisy_chunk_frames,
    noisy_observation_spans,
    resolve_actions_per_chunk,
    sink_read_chunks,
    teacher_forcing_frame_positions,
)

ROBOT_ROWS_NONE = "none"
ROBOT_ROWS_STATE = "state"
ROBOT_ROWS_ACTIONS = "actions"


def teacher_forcing_frame_segments(teacher_forcing_chunks: int, chunk_frames: int) -> torch.Tensor:
    """Return the segment of every latent frame of the teacher-forcing sequence: the sink is segment 0, clean chunk k segment 1 + k and noisy chunk c segment L + c."""
    noisy_frames = noisy_chunk_frames(chunk_frames)
    clean = torch.arange((teacher_forcing_chunks - 1) * chunk_frames, dtype=torch.long) // chunk_frames
    noisy = torch.arange(teacher_forcing_chunks * noisy_frames, dtype=torch.long) // noisy_frames
    return torch.cat([torch.zeros(1, dtype=torch.long), 1 + clean, teacher_forcing_chunks + noisy])


class SanaRobotAR(SanaRobotBackbone):
    """SANA-Robot-AR, the policy with memory.

    An episode is a sequence of action chunks. Every chunk denoises its future latent frames and its actions from
    its own observation latent and state, the first frame of the episode (the sink) and the memory of the chunks
    before it: the clean latents of the last ``sliding_window_chunks`` chunks in the softmax layers and a recurrent
    state over every past chunk in the Gated DeltaNet layers.

    Training runs a window of L chunks at once on the teacher-forcing sequence
    [sink | clean chunks 0..L-2 | noisy chunks 0..L-1] (``forward``). Inference streams the episode through per-layer
    caches: ``forward_sink`` opens them, and every chunk runs ``forward_observation``, the denoising steps of
    ``forward_chunk`` and one committing ``forward_chunk``.
    """

    def __init__(self, actions_per_chunk=32, sliding_window_chunks: int = 1, **backbone_arguments) -> None:
        """Build SANA-Robot-AR.

        Args:
            actions_per_chunk: Control steps per action chunk: an int, or a mapping from the control frequency
                (rounded, as a string) to an int.
            sliding_window_chunks: The window N: how many past chunks' clean latents the softmax layers read
                besides the sink.
            **backbone_arguments: Hyper-parameters of ``SanaRobotBackbone``.
        """
        super().__init__(**backbone_arguments)
        self.actions_per_chunk = actions_per_chunk
        self.sliding_window_chunks = sliding_window_chunks
        self.grad_checkpointing = False
        self._teacher_forcing_layout_cache = {}
        self._rope_table_cache = {}
        self._fps_cache = None

    def build_block(self, softmax: bool) -> PolicyBlock:
        """Build one layer of SANA-Robot-AR: softmax attention or a Gated DeltaNet, both causal across chunks, with per-chunk text cross-attention."""
        if softmax:
            self_attention = CausalSoftmaxAttention(
                self.hidden_size, self.hidden_size // self.softmax_head_dim
            )
        else:
            self_attention = CausalGatedDeltaNet(self.hidden_size, self.hidden_size // self.linear_head_dim)
        cross_attention = ChunkTextCrossAttention(self.hidden_size, self.cross_attention_heads)
        return PolicyBlock(self.hidden_size, self_attention, cross_attention, self.mlp_ratio)

    def forward(self, video_latents, timestep, text_embeddings, text_mask=None, data_info=None):
        """Predict the video flow and the action flow of a window of L chunks on the teacher-forcing sequence.

        Args:
            video_latents: Video latents [B, C, F, 1, sum(H_v * W_v)], every latent frame the concatenation of its
                flattened views. The frames are [sink | clean targets of chunks 0..L-2 | noisy chunks 0..L-1], a
                noisy chunk being its clean observation latent followed by its noisy target latents, so
                F = 1 + (L - 1) * f + L * (1 + f) for f target latents per chunk.
            timestep: Video timestep per latent frame [B, F]: 0 on the clean frames.
            text_embeddings: Text embeddings [B, V + 1, 1, L_text, C_text]: one group per view, then one for the
                robot rows.
            text_mask: Text token mask [B, V + 1, L_text], or None.
            data_info: Dict with ``teacher_forcing_num_chunks`` (L), ``view_latent_shape``, ``view_slot_ids`` (several
                views), ``model_fps`` and optionally ``video_frame_stride``; the robot rows of the L chunks, each
                chunk [state; its actions]: ``initial_state`` and ``initial_state_condition_mask`` (chunk 0's
                state), ``chunk_states`` and ``chunk_states_mask`` (the states of chunks 1..L-1), ``action`` and
                ``action_mask`` [B, L * (A + 1) - 1, 80] (the rows after the first state, the later states' rows
                being placeholders) and ``action_timestep`` for the same rows.

        Returns:
            Dict with ``video_pred``, the video flow in the layout of ``video_latents``, and ``action_pred`` in the
            layout of ``action``.
        """
        view_shapes = batch_view_shapes(data_info)
        view_areas = [height * width for (height, width) in view_shapes]
        (batch, device) = (video_latents.shape[0], video_latents.device)
        (chunk_actions, chunk_frames) = self.chunk_size(data_info)
        fps = self.batch_fps(data_info["model_fps"], batch, device)
        video_fps = fps / self.video_frame_stride(data_info)
        chunks = int(data_info["teacher_forcing_num_chunks"])

        self.set_latent_grid(video_latents)
        video_latents = video_latents.to(self.dtype)
        text_embeddings = text_embeddings.to(self.dtype)
        timestep = timestep.long().float()
        video = strip_to_view_tokens(self.x_embedder(video_latents), self.f, view_shapes)
        video_tokens = video.shape[1]
        robot = self._embed_teacher_forcing_rows(data_info, video, chunk_actions)
        robot_count = robot.shape[1]
        hidden = torch.cat((video, robot), dim=1)

        token_timestep = torch.cat(
            (
                strip_to_view_tokens(self.video_token_timesteps(timestep, video_tokens), self.f, view_shapes),
                timestep.new_zeros(batch, 1),
                self.action_row_timesteps(data_info, batch, robot_count - 1, device),
            ),
            dim=1,
        ).unsqueeze(1)
        time_embedding = self.embed_timesteps(token_timestep)
        modulation = self.t_block(time_embedding)

        frame_positions = teacher_forcing_frame_positions(
            chunks, chunk_frames, self.vae_temporal_stride, device=device
        )
        row = torch.arange(robot_count, dtype=torch.long, device=device)
        robot_steps = row - torch.div(row, chunk_actions + 1, rounding_mode="floor")
        rope_arguments = (data_info, view_shapes, fps, video_fps, frame_positions, robot_steps)
        rope_linear = self._rope(self.rope_linear, *rope_arguments)
        rope_softmax = self._rope(self.rope_softmax, *rope_arguments)
        sink_phases = self._sink_phases(chunks, chunk_frames, video_fps).to(rope_softmax.dtype)

        (text, text_mask) = self.embed_text(text_embeddings, text_mask)
        (order, restore, bounds) = self._teacher_forcing_layout(
            view_areas, robot_count, chunk_actions, chunk_frames, chunks, device
        )
        hidden = hidden.index_select(1, order)
        rope_linear = rope_linear.index_select(2, order)
        rope_softmax = rope_softmax.index_select(2, order)
        modulation = modulation.index_select(2, order)
        (text, text_spans) = self._pack_text(text, text_mask, bounds, view_areas, chunk_actions + 1, chunks)
        attention_arguments = {
            "chunk_bounds": bounds,
            "teacher_forcing_clean_segments": chunks,
            "observation_spans": noisy_observation_spans(view_areas, chunk_frames),
        }
        softmax_arguments = {"window_chunks": self.sliding_window_chunks, "sink_phases": sink_phases}
        hidden = self._run_teacher_forcing_layers(
            hidden,
            text,
            modulation,
            text_spans,
            rope_linear,
            rope_softmax,
            attention_arguments,
            softmax_arguments,
        )
        hidden = hidden.index_select(1, restore)

        video_hidden = self.final_layer(hidden[:, :video_tokens], time_embedding[:, :, :video_tokens])
        return {
            "video_pred": self.unpatchify(view_to_strip_tokens(video_hidden, self.f, view_shapes)),
            "action_pred": self.predict_actions(
                hidden[:, video_tokens + 1 :], time_embedding[:, :, video_tokens + 1 :], data_info
            ),
        }

    @torch.inference_mode()
    def forward_sink(
        self, video_latents, timestep, text_embeddings, text_mask=None, kv_caches=None, data_info=None
    ):
        """Open the caches of an episode with the sink, the latent of the episode's first frame.

        Args:
            video_latents: The sink latent [B, C, 1, 1, sum(H_v * W_v)], the first frame of every view encoded
                alone.
            timestep: Video timestep, 0.
            text_embeddings: Text embeddings [B, V + 1, 1, L_text, C_text].
            text_mask: Text token mask, or None.
            kv_caches: The per-layer caches, empty: a dict per layer, in which a Gated DeltaNet layer keeps its
                state under ``"S"`` and a softmax layer its key/value cache manager under ``"kv"``.
            data_info: Dict with ``view_latent_shape``, ``view_slot_ids`` (several views) and ``model_fps``.

        Returns:
            Dict with ``video_pred``, the prediction for the sink tokens (unused).
        """
        return self._forward_window(
            video_latents,
            timestep,
            text_embeddings,
            text_mask,
            data_info,
            robot_rows=ROBOT_ROWS_NONE,
            kv_caches=list(kv_caches),
            save_kv_cache=True,
            step_offset=0,
            frame_offset=0,
        )

    @torch.inference_mode()
    def forward_observation(
        self,
        video_latents,
        timestep,
        text_embeddings,
        text_mask=None,
        kv_caches=None,
        chunk_offset=None,
        chunk_offset_tensor=None,
        data_info=None,
    ):
        """Read the observation of chunk ``chunk_offset`` into the caches, before the chunk's denoising steps.

        The observation latent and the state row read the sink, the memory and each other. Nothing is committed:
        the layers keep what the observation wrote next to the memory, the denoising steps of this chunk read
        it, and the next commit drops it.

        Args:
            video_latents: The observation latent [B, C, 1, 1, sum(H_v * W_v)], the chunk's first frame of every
                view encoded alone.
            timestep: Video timestep, 0.
            text_embeddings: Text embeddings [B, V + 1, 1, L_text, C_text].
            text_mask: Text token mask, or None.
            kv_caches: The per-layer caches after the sink and the chunks before this one.
            chunk_offset: Index of the chunk in the episode, from 0.
            chunk_offset_tensor: ``chunk_offset`` as a device scalar; when given, the positions derive from it.
            data_info: Dict with ``view_latent_shape``, ``view_slot_ids`` (several views), ``model_fps`` and the
                chunk's state, ``initial_state`` and ``initial_state_condition_mask``.

        Returns:
            Dict with ``video_pred``, the prediction for the observation tokens (unused).
        """
        (chunk_actions, chunk_frames) = self.chunk_size(data_info)
        chunk = int(chunk_offset) if chunk_offset_tensor is None else chunk_offset_tensor
        return self._forward_window(
            video_latents,
            timestep,
            text_embeddings,
            text_mask,
            data_info,
            robot_rows=ROBOT_ROWS_STATE,
            kv_caches=list(kv_caches),
            observation_prefill=True,
            step_offset=chunk * chunk_actions,
            frame_offset=chunk * chunk_frames,
        )

    @torch.inference_mode()
    def forward_chunk(
        self,
        video_latents,
        timestep,
        text_embeddings,
        text_mask=None,
        kv_caches=None,
        save_kv_cache=False,
        chunk_offset=None,
        chunk_offset_tensor=None,
        data_info=None,
    ):
        """Run one denoising step of chunk ``chunk_offset`` against the caches, or commit the chunk to them.

        A denoising step (``save_kv_cache`` False) runs the noisy target latents and the noisy action rows of the
        chunk; they read the sink, the memory and the chunk's observation, which ``forward_observation`` must have
        read first. A commit (``save_kv_cache`` True) runs the clean target latents of the chunk without robot
        rows and writes them into the caches as memory.

        Args:
            video_latents: The chunk's target latents [B, C, f, 1, sum(H_v * W_v)]: noisy for a denoising step,
                clean for a commit.
            timestep: Video timestep: the step's timestep, or 0 for a commit.
            text_embeddings: Text embeddings [B, V + 1, 1, L_text, C_text].
            text_mask: Text token mask, or None.
            kv_caches: The per-layer caches.
            save_kv_cache: False for a denoising step, True for a commit.
            chunk_offset: Index of the chunk in the episode, from 0.
            chunk_offset_tensor: ``chunk_offset`` as a device scalar; when given, the positions derive from it.
            data_info: Dict with ``view_latent_shape``, ``view_slot_ids`` (several views) and ``model_fps``; for a
                denoising step also the noisy actions ``action`` [B, A, 80], ``action_mask`` and
                ``action_timestep``.

        Returns:
            Dict with ``video_pred``, the video flow of the target latents, and for a denoising step
            ``action_pred`` [B, A, 80].
        """
        (chunk_actions, chunk_frames) = self.chunk_size(data_info)
        chunk = int(chunk_offset) if chunk_offset_tensor is None else chunk_offset_tensor
        return self._forward_window(
            video_latents,
            timestep,
            text_embeddings,
            text_mask,
            data_info,
            robot_rows=ROBOT_ROWS_NONE if save_kv_cache else ROBOT_ROWS_ACTIONS,
            kv_caches=list(kv_caches),
            save_kv_cache=bool(save_kv_cache),
            step_offset=chunk * chunk_actions,
            frame_offset=chunk * chunk_frames + latent_centre_offset(self.vae_temporal_stride),
        )

    def _forward_window(
        self,
        video_latents,
        timestep,
        text_embeddings,
        text_mask,
        data_info,
        *,
        robot_rows: str,
        kv_caches,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
        step_offset=0,
        frame_offset=0,
    ):
        """Run one streaming window of a single chunk against the caches. ``step_offset`` and ``frame_offset`` place the window's robot rows and latent frames on the episode clock."""
        view_shapes = batch_view_shapes(data_info)
        view_areas = [height * width for (height, width) in view_shapes]
        (batch, device) = (video_latents.shape[0], video_latents.device)
        (chunk_actions, chunk_frames) = self.chunk_size(data_info)
        fps = self.batch_fps(data_info["model_fps"], batch, device)
        video_fps = fps / self.video_frame_stride(data_info)

        self.set_latent_grid(video_latents)
        video_latents = video_latents.to(self.dtype)
        text_embeddings = text_embeddings.to(self.dtype)
        timestep = timestep.long().float()
        video = strip_to_view_tokens(self.x_embedder(video_latents), self.f, view_shapes)
        video_tokens = video.shape[1]
        robot = self._embed_robot_rows(data_info, video, robot_rows)
        robot_count = 0 if robot is None else robot.shape[1]
        hidden = video if robot is None else torch.cat((video, robot), dim=1)

        timesteps = [
            strip_to_view_tokens(self.video_token_timesteps(timestep, video_tokens), self.f, view_shapes)
        ]
        if robot_rows == ROBOT_ROWS_STATE:
            timesteps.append(timesteps[0].new_zeros(batch, 1))
        if robot_rows == ROBOT_ROWS_ACTIONS:
            timesteps.append(self.action_row_timesteps(data_info, batch, robot_count, device))
        token_timestep = torch.cat(timesteps, dim=1).unsqueeze(1)
        time_embedding = self.embed_timesteps(token_timestep)
        modulation = self.t_block(time_embedding)

        frame_positions = frame_offset + torch.arange(self.f, dtype=torch.float64, device=device)
        robot_steps = None
        if robot_count:
            row = torch.arange(robot_count, dtype=torch.long, device=device) + int(
                robot_rows == ROBOT_ROWS_ACTIONS
            )
            robot_steps = step_offset + row - torch.div(row, chunk_actions + 1, rounding_mode="floor")
        rope_linear = self._rope(
            self.rope_linear, data_info, view_shapes, fps, video_fps, frame_positions, robot_steps
        )
        managers = [kv_caches[layer]["kv"] for layer in self.softmax_layer_indices]
        tables = self._rope_tables(
            data_info, view_shapes, fps, video_fps, chunk_actions, chunk_frames, managers[0].clock_chunks
        )
        for manager in managers:
            manager.rope_tables = tables

        (text, text_mask) = self.embed_text(text_embeddings, text_mask)
        segment_robot_rows = {ROBOT_ROWS_STATE: 1, ROBOT_ROWS_ACTIONS: chunk_actions}
        (text, text_spans) = self._pack_text(
            text,
            text_mask,
            [0, hidden.shape[1]],
            view_areas,
            segment_robot_rows.get(robot_rows, 0),
            int(robot_rows == ROBOT_ROWS_NONE),
        )
        attention_arguments = {"save_kv_cache": save_kv_cache, "observation_prefill": observation_prefill}
        hidden = self._run_layers(
            hidden, text, modulation, text_spans, rope_linear, attention_arguments, kv_caches
        )

        video_hidden = self.final_layer(hidden[:, :video_tokens], time_embedding[:, :, :video_tokens])
        result = {"video_pred": self.unpatchify(view_to_strip_tokens(video_hidden, self.f, view_shapes))}
        if robot_rows == ROBOT_ROWS_ACTIONS:
            result["action_pred"] = self.predict_actions(
                hidden[:, video_tokens:], time_embedding[:, :, video_tokens:], data_info
            )
        return result

    def chunk_size(self, data_info: dict) -> tuple[int, int]:
        """Return the control steps and the target latent frames of one chunk at the batch's control frequency."""
        chunk_actions = resolve_actions_per_chunk(
            self.actions_per_chunk, self._host_fps(data_info["model_fps"])
        )
        return (
            chunk_actions,
            chunk_actions // (self.vae_temporal_stride * self.video_frame_stride(data_info)),
        )

    def _host_fps(self, model_fps) -> float:
        """Return the control frequency as a Python float, remembered per source tensor."""
        if not torch.is_tensor(model_fps):
            return float(model_fps)
        key = (model_fps.data_ptr(), tuple(model_fps.shape), str(model_fps.device), str(model_fps.dtype))
        if self._fps_cache is not None and self._fps_cache[0] == key:
            return self._fps_cache[1]
        value = float(model_fps.reshape(-1)[0].item())
        self._fps_cache = (key, value, model_fps)
        return value

    def _embed_robot_rows(self, data_info: dict, reference: torch.Tensor, robot_rows: str):
        """Return the robot tokens of the window: none, the chunk's state row or its action rows."""
        if robot_rows == ROBOT_ROWS_NONE:
            return None
        (device, dtype) = (reference.device, reference.dtype)
        if robot_rows == ROBOT_ROWS_STATE:
            state = data_info["initial_state"].to(device)
            state_mask = data_info["initial_state_condition_mask"].to(device)
            return self.state_embed(state.to(dtype=dtype), state_mask).unsqueeze(1)
        action = data_info["action"].to(device)
        return self.action_embed(action.to(dtype=dtype), data_info["action_mask"].to(device))

    def _embed_teacher_forcing_rows(self, data_info: dict, reference: torch.Tensor, chunk_actions: int):
        """Return the robot tokens of the teacher-forcing sequence: every chunk's [state; actions]."""
        (device, dtype) = (reference.device, reference.dtype)
        state = data_info["initial_state"].to(device)
        state_mask = data_info["initial_state_condition_mask"].to(device)
        state_token = self.state_embed(state.to(dtype=dtype), state_mask).unsqueeze(1)
        action = data_info["action"].to(device)
        action_tokens = self.action_embed(action.to(dtype=dtype), data_info["action_mask"].to(device))
        robot = torch.cat((state_token.to(dtype=action_tokens.dtype), action_tokens), dim=1)
        chunks = robot.shape[1] // (chunk_actions + 1)
        if chunks > 1:
            batch = robot.shape[0]
            later_states = data_info["chunk_states"].to(device).reshape(batch * (chunks - 1), ROBOT_DIM)
            later_masks = data_info["chunk_states_mask"].to(device).reshape(batch * (chunks - 1), ROBOT_DIM)
            later_tokens = self.state_embed(later_states.to(dtype=dtype), later_masks).reshape(
                batch, chunks - 1, -1
            )
            state_rows = torch.arange(1, chunks, device=device) * (chunk_actions + 1)
            robot = robot.index_copy(1, state_rows, later_tokens.to(dtype=robot.dtype))
        return robot

    def _sink_phases(self, teacher_forcing_chunks: int, chunk_frames: int, video_fps) -> torch.Tensor:
        """Return the rotary phases that move the sink from time 0 to the time a softmax window reads it at, one per window start: a reader never sees the sink farther than ``SINK_HORIZON_CHUNKS`` chunks in the past."""
        sink_chunk = sink_read_chunks(
            teacher_forcing_chunks, self.sliding_window_chunks, device=video_fps.device
        )
        time = (
            self.rope_softmax.base_fps
            * (sink_chunk.to(torch.float64) * chunk_frames)[None]
            / video_fps[:, None]
        )
        position_ids = torch.stack((time, torch.zeros_like(time), torch.zeros_like(time)), dim=-1)
        return self.rope_softmax.from_position_ids(position_ids)

    def _teacher_forcing_layout(
        self, view_areas, robot_count, chunk_actions, chunk_frames, teacher_forcing_chunks, device
    ):
        """Return (order, restore, bounds) of the teacher-forcing sequence: the permutation that gathers every segment's tokens as [view 1; ...; view V; robot rows], its inverse, and the segment bounds [sink | clean chunks | noisy chunks]."""
        key = (
            self.f,
            tuple(view_areas),
            robot_count,
            chunk_actions,
            chunk_frames,
            teacher_forcing_chunks,
            str(device),
        )
        cached = self._teacher_forcing_layout_cache.get(key)
        if cached is not None:
            return cached
        frame_segment = teacher_forcing_frame_segments(teacher_forcing_chunks, chunk_frames)
        video_segment = torch.cat([frame_segment.repeat_interleave(area) for area in view_areas])
        video_group = torch.cat(
            [torch.full((self.f * area,), view, dtype=torch.long) for (view, area) in enumerate(view_areas)]
        )
        robot_segment = (
            torch.arange(robot_count, dtype=torch.long) // (chunk_actions + 1) + teacher_forcing_chunks
        )
        robot_group = torch.full((robot_count,), len(view_areas), dtype=torch.long)
        segment = torch.cat([video_segment, robot_segment])
        group = torch.cat([video_group, robot_group])
        order = torch.argsort(segment * (len(view_areas) + 1) + group, stable=True)
        counts = torch.bincount(
            segment, minlength=robot_count // (chunk_actions + 1) + teacher_forcing_chunks
        )
        bounds = [0] + torch.cumsum(counts, dim=0).tolist()
        if len(self._teacher_forcing_layout_cache) >= 64:
            self._teacher_forcing_layout_cache.clear()
        layout = (order.to(device), torch.argsort(order).to(device), bounds)
        self._teacher_forcing_layout_cache[key] = layout
        return layout

    def _rope(self, rope, data_info: dict, view_shapes: tuple, fps, video_fps, frame_positions, robot_steps):
        """Return the rotary phases of every token on the episode clock: latent frame j at ``frame_positions[j]`` latent frames, a robot row at its control step ``robot_steps`` (the state at the step its chunk starts)."""
        device = fps.device
        video_ids = self.video_position_ids(rope, data_info, view_shapes, video_fps, self.f, device).clone()
        token_frame = torch.cat(
            [
                torch.arange(self.f, device=device).repeat_interleave(height * width)
                for (height, width) in view_shapes
            ]
        )
        video_ids[..., 0] = rope.base_fps * frame_positions[token_frame][None] / video_fps[:, None]
        video = rope.from_position_ids(video_ids)
        if robot_steps is None:
            return video
        robot_time = (
            rope.base_fps * robot_steps.to(torch.float64)[None] / (self.vae_temporal_stride * fps[:, None])
        )
        robot_ids = torch.stack(
            (robot_time, torch.zeros_like(robot_time), torch.zeros_like(robot_time)), dim=-1
        )
        return torch.cat((video, rope.from_position_ids(robot_ids).to(dtype=video.dtype)), dim=2)

    def _rope_tables(self, data_info, view_shapes, fps, video_fps, chunk_actions, chunk_frames, clock_chunks):
        """Return the separable rotary phase tables (time, spatial, robot) from which the softmax cache managers assemble the phases of a streaming read, covering ``clock_chunks`` chunks."""
        rope = self.rope_softmax
        device = fps.device
        slots = tuple(batch_view_slot_ids(data_info, len(view_shapes))) if len(view_shapes) > 1 else ()
        key = (self._host_fps(data_info["model_fps"]), view_shapes, slots, chunk_frames, str(device))
        cached = self._rope_table_cache.get(key)
        if cached is not None and cached[0] >= int(clock_chunks):
            return cached[1]
        frames = 1 + chunk_frames * (int(clock_chunks) + 1) + 1
        steps = chunk_actions * (int(clock_chunks) + 1) + 1 + chunk_actions // chunk_frames
        (fps, video_fps) = (fps[:1], video_fps[:1])
        units = 2 * self.vae_temporal_stride
        frame_time = torch.arange(frames * units, dtype=torch.float64, device=device) / units
        video_time = (rope.base_fps * frame_time[None] / video_fps[:, None])[0]
        step = torch.arange(steps, dtype=torch.float64, device=device)
        robot_time = (rope.base_fps * step[None] / (self.vae_temporal_stride * fps[:, None]))[0]
        spatial_ids = self.video_position_ids(rope, data_info, view_shapes, video_fps, 1, device)[0]

        def axis_phase(positions, axis):
            dim = rope.axis_dims[axis]
            exponent = torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim
            phase = positions.to(torch.float64)[:, None] * rope.theta ** (-exponent)
            return torch.polar(torch.ones_like(phase), phase)

        spatial = torch.cat([axis_phase(spatial_ids[:, 1], 1), axis_phase(spatial_ids[:, 2], 2)], dim=-1)
        robot = torch.cat([axis_phase(robot_time, 0), torch.ones_like(spatial[:1]).expand(steps, -1)], dim=-1)
        tables = {"time": axis_phase(video_time, 0), "spatial": spatial, "robot": robot}
        self._rope_table_cache[key] = (int(clock_chunks), tables)
        return tables

    def _pack_text(self, text, text_mask, bounds, view_areas, segment_robot_rows, robotless_segments):
        """Pack the grouped text for per-chunk cross-attention: every (segment, group) token span reads the text of its group. Returns the packed text tokens [1, T, hidden] and the ``ChunkTextSpans``."""
        (batch, groups) = (text.shape[0], text.shape[1])
        frame_area = sum(view_areas)
        segment_spans = []
        for segment in range(len(bounds) - 1):
            tokens = bounds[segment + 1] - bounds[segment]
            if segment < robotless_segments:
                segment_spans.append([tokens // frame_area * area for area in view_areas])
            else:
                frames = (tokens - segment_robot_rows) // frame_area
                segment_spans.append([frames * area for area in view_areas] + [segment_robot_rows])
        query_lengths = [length for _ in range(batch) for spans in segment_spans for length in spans]
        group_text = [
            [text[sample, group][text_mask[sample, group].to(torch.bool)] for group in range(groups)]
            for sample in range(batch)
        ]
        (pieces, text_lengths) = ([], [])
        for sample in range(batch):
            for spans in segment_spans:
                for group in range(len(spans)):
                    pieces.append(group_text[sample][group])
                    text_lengths.append(int(group_text[sample][group].shape[0]))
        packed_text = torch.cat(pieces, dim=0).unsqueeze(0)
        return (packed_text, chunk_text_spans(query_lengths, text_lengths, text.device))

    def _run_layers(self, hidden, text, modulation, text_spans, rope_linear, attention_arguments, kv_caches):
        """Run the layer stack; every finished group of ``attn_res_block_size`` layers becomes an AttnRes source."""
        attn_res = self.attn_res
        depth = len(self.blocks)
        sources = [hidden]
        for first_layer in range(0, depth, self.attn_res_block_size):
            partial_block = None
            for layer in range(first_layer, min(first_layer + self.attn_res_block_size, depth)):
                layer_arguments = dict(
                    attention_arguments,
                    rotary_emb=None if layer in self.softmax_layer_indices else rope_linear,
                    kv_cache=kv_caches[layer],
                )
                attention_input = attn_res.attend(attn_res.attn_proj, sources, partial_block)
                attention_delta = self.blocks[layer](
                    attention_input,
                    text,
                    modulation,
                    {"text_spans": text_spans},
                    sublayer="attention",
                    **layer_arguments,
                )
                partial_block = attention_delta if partial_block is None else partial_block + attention_delta
                mlp_input = attn_res.attend(attn_res.mlp_proj, sources, partial_block)
                partial_block = partial_block + self.blocks[layer](mlp_input, modulation, sublayer="mlp")
            sources.append(partial_block)
        return attn_res.attend(attn_res.final_proj, sources)

    def _run_teacher_forcing_layers(
        self,
        hidden,
        text,
        modulation,
        text_spans,
        rope_linear,
        rope_softmax,
        attention_arguments,
        softmax_arguments,
    ):
        """Run the layer stack on the teacher-forcing sequence; every finished group of ``attn_res_block_size`` layers becomes an AttnRes source. Training computes the mixing logits of the finished sources once per forward and recomputes every layer in the backward pass when ``grad_checkpointing`` is set."""
        attn_res = self.attn_res
        depth = len(self.blocks)
        cache_logits = self.training
        checkpoint = self.grad_checkpointing and self.training
        sources = [hidden]
        (attention_logits, mlp_logits) = ([], [])
        if cache_logits:
            attention_logits.append(attn_res.source_logit(attn_res.attn_proj, hidden))
            mlp_logits.append(attn_res.source_logit(attn_res.mlp_proj, hidden))
        for first_layer in range(0, depth, self.attn_res_block_size):
            partial_block = None
            for layer in range(first_layer, min(first_layer + self.attn_res_block_size, depth)):
                softmax = layer in self.softmax_layer_indices
                layer_arguments = dict(
                    attention_arguments,
                    rotary_emb=rope_softmax if softmax else rope_linear,
                    **(softmax_arguments if softmax else {}),
                )
                step = (
                    attn_res,
                    self.blocks[layer],
                    tuple(sources),
                    partial_block,
                    text,
                    modulation,
                    text_spans,
                    layer_arguments,
                    tuple(attention_logits) if cache_logits else None,
                    tuple(mlp_logits) if cache_logits else None,
                )
                if checkpoint:
                    (attention_delta, mlp_delta) = torch_checkpoint(_layer_step, *step, use_reentrant=False)
                else:
                    (attention_delta, mlp_delta) = _layer_step(*step)
                partial_block = attention_delta if partial_block is None else partial_block + attention_delta
                partial_block = partial_block + mlp_delta
            sources.append(partial_block)
            if cache_logits and first_layer + self.attn_res_block_size < depth:
                attention_logits.append(attn_res.source_logit(attn_res.attn_proj, partial_block))
                mlp_logits.append(attn_res.source_logit(attn_res.mlp_proj, partial_block))
        if cache_logits:
            final_logits = [attn_res.source_logit(attn_res.final_proj, source) for source in sources]
            return attn_res.attend_cached(attn_res.final_proj, sources, final_logits)
        return attn_res.attend(attn_res.final_proj, sources)


def _layer_step(
    attn_res,
    block,
    sources,
    partial_block,
    text,
    modulation,
    text_spans,
    layer_arguments,
    attention_logits,
    mlp_logits,
):
    """Run both sublayers of one layer on their AttnRes inputs and return the two residual updates. Every input arrives as an argument so that gradient checkpointing replays the step exactly."""
    sources = list(sources)

    def attend(projection, logits, partial):
        if logits is None:
            return attn_res.attend(projection, sources, partial)
        return attn_res.attend_cached(projection, sources, list(logits), partial)

    attention_input = attend(attn_res.attn_proj, attention_logits, partial_block)
    attention_delta = block(
        attention_input, text, modulation, {"text_spans": text_spans}, sublayer="attention", **layer_arguments
    )
    partial_block = attention_delta if partial_block is None else partial_block + attention_delta
    mlp_input = attend(attn_res.mlp_proj, mlp_logits, partial_block)
    return (attention_delta, block(mlp_input, modulation, sublayer="mlp"))
