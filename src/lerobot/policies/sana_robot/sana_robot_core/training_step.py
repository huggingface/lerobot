# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/train/step/ar.py. Import paths are rewritten to the LeRobot package;
# the inputs, the targets and the loss of one window; the latent store, the padded batch of several windows and its packed row are not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch

from lerobot.policies.sana_robot.sana_robot_core.flow_loss import (
    chunk_action_losses,
    chunk_video_losses,
    mean_over_trained_chunks,
)
from lerobot.policies.sana_robot.sana_robot_core.flow_noise import FlowNoise, add_noise
from lerobot.policies.sana_robot.sana_robot_core.flow_timesteps import (
    ACTION_FLOW_SHIFT,
    VIDEO_FLOW_SHIFT,
    model_timesteps,
    sample_chunk_timesteps,
    shifted_sigmas,
)
from lerobot.policies.sana_robot.sana_robot_core.flow_weights import timestep_weights
from lerobot.policies.sana_robot.sana_robot_core.teacher_forcing_sequence import (
    action_row_positions,
    assemble_teacher_forcing_latents,
    target_frame_positions,
    teacher_forcing_action_timesteps,
    teacher_forcing_video_timesteps,
    with_state_rows,
)

POLICY_FIELDS = (
    "view_latent_shape",
    "view_slot_ids",
    "model_fps",
    "video_frame_stride",
    "initial_state",
    "initial_state_condition_mask",
    "chunk_states",
    "chunk_states_mask",
    "action_mask",
)


def draw_window_noise(window_latents: torch.Tensor, chunks: int, chunk_size: int, slots: int) -> FlowNoise:
    """Draw the timesteps and the noise of a training step for a batch of windows of ``chunks`` chunks."""
    batch, device = window_latents.shape[0], window_latents.device
    timesteps = sample_chunk_timesteps(batch, int(chunks), device)
    video = torch.randn_like(window_latents[:, :, 1:])
    action = torch.randn(batch, int(chunks) * int(chunk_size), int(slots), device=device)
    return FlowNoise(timesteps, video, action)


def window_flow_inputs(
    batch: dict,
    window_latents: torch.Tensor,
    observation_latents: torch.Tensor,
    noise: FlowNoise,
    timestep_weighting: bool = True,
) -> tuple[dict, dict]:
    """Build the policy inputs and the flow targets of a training step.

    Args:
        batch: A batch of ``EpisodeWindowDataset`` on the training device.
        window_latents: The causal encoding of every window [B, C, 1 + L * f, tokens].
        observation_latents: The observations of chunks 1 to L - 1 [B, C, L - 1, tokens].
        noise: The draws of the step.
        timestep_weighting: Weigh the loss of a chunk by its timestep; every chunk weighs 1 when off.

    Returns:
        The keyword arguments of ``SanaRobotAR.forward`` without the text, and a dict with the flow targets
        ``video`` [B, C, L * f, tokens] and ``action`` [B, L * A, 80], the loss weights ``video_weights`` [B, L * f]
        and ``action_weights`` [B, L * A], and ``video_positions`` and ``action_positions``, where the targets sit
        in the policy's outputs.
    """
    chunks = int(batch["teacher_forcing_num_chunks"])
    per_chunk = (window_latents.shape[2] - 1) // chunks
    chunk_size = noise.action.shape[1] // chunks
    device = window_latents.device
    video_positions = target_frame_positions(chunks, per_chunk).to(device)
    action_positions = action_row_positions(chunks, chunk_size).to(device)
    frame_timesteps = noise.timesteps.repeat_interleave(per_chunk, dim=1)
    row_timesteps = noise.timesteps.repeat_interleave(chunk_size, dim=1)

    video_sigmas = shifted_sigmas(frame_timesteps, VIDEO_FLOW_SHIFT)[:, None, :, None]
    noisy_video, video_target = add_noise(window_latents[:, :, 1:], noise.video, video_sigmas)
    latents = assemble_teacher_forcing_latents(window_latents, observation_latents, chunks)
    latents[:, :, video_positions] = noisy_video

    action_sigmas = shifted_sigmas(row_timesteps, ACTION_FLOW_SHIFT)[:, :, None]
    noisy_action, action_target = add_noise(batch["action"][:, action_positions], noise.action, action_sigmas)
    read_slots = batch["action_mask"][:, action_positions]
    noisy_action = torch.where(read_slots, noisy_action, torch.zeros_like(noisy_action))

    data_info = {name: batch[name] for name in POLICY_FIELDS}
    data_info["teacher_forcing_num_chunks"] = chunks
    data_info["action"] = with_state_rows(noisy_action, chunks, chunk_size)
    data_info["action_timestep"] = teacher_forcing_action_timesteps(
        model_timesteps(noise.timesteps, ACTION_FLOW_SHIFT), chunk_size
    )
    video_weights = timestep_weights(frame_timesteps, VIDEO_FLOW_SHIFT)
    action_weights = timestep_weights(row_timesteps, ACTION_FLOW_SHIFT)
    if not timestep_weighting:
        video_weights, action_weights = torch.ones_like(video_weights), torch.ones_like(action_weights)
    inputs = {
        "video_latents": latents.unsqueeze(3),
        "timestep": teacher_forcing_video_timesteps(
            model_timesteps(noise.timesteps, VIDEO_FLOW_SHIFT), per_chunk
        ),
        "data_info": data_info,
    }
    targets = {
        "video": video_target,
        "action": action_target,
        "video_weights": video_weights,
        "action_weights": action_weights,
        "video_positions": video_positions,
        "action_positions": action_positions,
    }
    return inputs, targets


def window_losses(output: dict, targets: dict, batch: dict, action_loss_weight: float) -> dict:
    """Compute the training loss of a batch of windows from the policy's output.

    Args:
        output: The output of ``SanaRobotAR.forward``.
        targets: The targets from ``window_flow_inputs``.
        batch: The batch of the step, for ``chunk_supervised`` and ``action_loss_mask``.
        action_loss_weight: The weight of the action loss in the total.

    Returns:
        A dict with ``loss``, the mean over the trained chunks of the batch, its parts ``video_loss`` and
        ``action_loss``, and ``trained``, the trained chunks [B, L].
    """
    trained = batch["chunk_supervised"].bool()
    chunks = trained.shape[1]
    per_chunk = targets["video"].shape[2] // chunks
    video_prediction = output["video_pred"].squeeze(3)[:, :, targets["video_positions"]]
    action_prediction = output["action_pred"][:, targets["action_positions"]]
    video_losses, video_counts = chunk_video_losses(
        video_prediction,
        targets["video"],
        targets["video_weights"],
        trained.repeat_interleave(per_chunk, dim=1),
        chunks,
    )
    action_losses, action_counts = chunk_action_losses(
        action_prediction,
        targets["action"],
        targets["action_weights"],
        batch["action_loss_mask"][:, targets["action_positions"]],
        chunks,
    )
    video_loss = mean_over_trained_chunks(video_losses, trained & (video_counts > 0))
    action_loss = mean_over_trained_chunks(action_losses, trained & (action_counts > 0))
    return {
        "loss": video_loss + action_loss_weight * action_loss,
        "video_loss": video_loss,
        "action_loss": action_loss,
        "trained": trained,
    }
