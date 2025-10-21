# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import threading
import time
from collections import deque
from queue import Empty, Queue

import torch
from torch import Tensor

from lerobot.policies.rtc_smolvla.configuration_rtc_smolvla import RTCSmolVLA
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, make_att_2d_masks
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

logger = logging.getLogger(__name__)


class VLAFlowMatchingRealTimeCorrected(VLAFlowMatching):
    """
    Real-time chunking wrapper around a (frozen) flow-matching VLA.
    This class runs a background inference loop and performs *guided inpainting*
    using the previous chunk per Algorithm 1 (Sec. 4.3) with:
      • denoising function f(A) = A + (1 - τ) vπ(A, o, τ)  (Eq. 3)
      • guidance term via VJP: J^T [ diag(W) (A_prev - f(A)) ] (Eq. 2)
      • Euler step: A <- A + (1/n) [ vπ + clip(β) * guidance ] (Eq. 1)
      • soft mask W over indices i = 0..H-1 (Eq. 5)
    """

    def __init__(self, config):
        super().__init__(config)

        # Horizons & hyperparameters (paper notation in comments)
        self.horizon = self.config.chunk_size  # H
        self.denoising_steps = self.config.num_steps  # n
        self.beta = self.config.beta if self.config.beta else 5  # β (recommended 5 in paper)

        # Background I/O
        self.input_queue: Queue[tuple] = Queue()  # (images, img_masks, lang_tokens, lang_masks, state, s, d)
        self.output_queue: Queue[Tensor] = Queue()  # A_new of shape [B, H, action_dim]

        self.device = self.config.device

        # Freeze the backbone with expert for inference-time inpainting
        for p in self.vlm_with_expert.parameters():
            p.requires_grad = False

        # Keep track of the *last* chunk we produced (A_cur in Alg. 1)
        self.current_action: Tensor | None = None

    def start_inference_loop(self):
        """Spawn the background inference loop (Alg. 1: INFERENCELOOP)."""
        thread = threading.Thread(target=self._inference_loop, daemon=True)
        thread.start()

    # Runs at max speed; the *controller cadence* is handled by the policy.
    def _inference_loop(self):
        while True:
            # Controller provides latest obs and scheduling info (s, d)
            images, img_masks, lang_tokens, lang_masks, state, start_horizon, delay = self.input_queue.get()

            images = [img.clone() for img in images]
            img_masks = [mask.clone() for mask in img_masks]
            lang_tokens = lang_tokens.clone()
            lang_masks = lang_masks.clone()
            state = state.clone()

            if self.current_action is None:
                # First time: unguided flow integration A ~ π(·|o) (no inpainting).
                t0 = time.time()
                self.current_action = self.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state
                )  # [B, H, action_dim]
                logger.debug(f"[RTC] Initial unguided sample time: {time.time() - t0:.3f}s")
            else:
                # Subsequent calls: guided inpainting per Algorithm 1 (lines 23–29)
                # Build A_prev by dropping the first s executed actions (line 15).
                previous_action = self.current_action[:, start_horizon:, :]  # [B, H - s, D]
                t0 = time.time()
                a_new = self.guided_inference(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    previous_action=previous_action,
                    delay=delay,  # d = max(Q) (line 17)
                    start_horizon=start_horizon,  # s = actions executed since last start (line 14)
                )  # returns [B, H, D]
                logger.debug(f"[RTC] Guided inpainting time: {time.time() - t0:.3f}s")
                self.current_action = a_new

            # Publish the *entire* new chunk A_cur (not shifted/padded).
            self.output_queue.put(self.current_action)

    # Core guided inpainting (Algorithm 1: GUIDEDINFERENCE)
    def guided_inference(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        previous_action: Tensor,
        delay: int,
        start_horizon: int,
    ) -> Tensor:
        """
        Args:
          previous_action: A_prev = A_cur[s: H], shape [B, H - s, D]
          delay:          d = max(Q), predicted inference delay in ticks
          start_horizon:  s = # actions executed since last chunk start

        Returns:
          A_new: new chunk of shape [B, H, D]
        """
        bsize = state.shape[0]
        device = state.device
        state = state.clone().detach()

        # Prefix embed + KV cache over (images, language, state)
        # This follows the standard "prefix once, suffix many" pattern to speed up
        # the iterative denoising passes. (Not explicit in Alg. 1 but matches Eq. 1–3 usage.)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # Right-pad A_prev to length H (Algorithm 1, line 24)
        horizon = self.horizon
        dimension = self.config.max_action_dim
        if previous_action.shape[1] < horizon:
            pad_len = horizon - previous_action.shape[1]
            previous_action = torch.cat(
                [previous_action, torch.zeros(bsize, pad_len, previous_action.shape[2], device=device)], dim=1
            )  # [B, H, D]

        #  Build soft mask W (Eq. 5) over i = 0..H-1
        # W_i = 1                   if i < d
        # W_i = c_i * e^{c_i - 1}/(e - 1)   if d <= i < H - s   where c_i = (H - s - i) / (H - s - d + 1)
        # W_i = 0                   if i >= H - s
        weights = self.soft_mask_exact(h=horizon, s=start_horizon, d=delay, device=device)  # [1, H, 1]

        # Initialize noise A^0 ~ N(0, I) and integrate for n steps (Eq. 1)
        a_t = self.sample_noise((bsize, horizon, dimension), device)  # A^0  (Algorithm 1, line 24)
        n = self.denoising_steps
        dt = 1.0 / n

        eps = 1e-8  # numerical stability

        # Denoising loop with ΠGDM guidance (Alg. 1 lines 25–29)
        for i in range(n):
            tau = 1.0 * i / n  # τ ∈ [0,1) increasing
            tau_tensor = torch.full((bsize,), tau, device=device)

            # Define f(A) = A + (1 - τ) vπ(A, o, τ)  (Eq. 3)
            def f_denoise(a_in: Tensor, tau=tau, tau_tensor=tau_tensor) -> Tensor:
                v = self.denoise_step(prefix_pad_masks, past_key_values, a_in, tau_tensor)
                return a_in + (1.0 - tau) * v

            with torch.enable_grad():
                a_t = a_t.requires_grad_(True)

                # vπ(A_t, o, τ)
                v_t = self.denoise_step(prefix_pad_masks, past_key_values, a_t, tau_tensor)  # [B, H, D]

                # f(A_t) and weighted error e = diag(W) (A_prev - f(A_t))   (Alg. 1, line 27)
                f_at = a_t + (1.0 - tau) * v_t
                e = (previous_action - f_at) * weights  # [B, H, D]

                # Vector-Jacobian product g = J^T e  (Alg. 1, line 28; Eq. 2)
                # Use reverse-mode autodiff to compute VJP of f_denoise at A_t against e
                _, g = torch.autograd.functional.vjp(f_denoise, a_t, v=e, create_graph=False)

                # r_τ^2 = (1 - τ)^2 / (τ^2 + (1 - τ)^2)   (Eq. 4)
                r2 = ((1.0 - tau) ** 2) / (tau**2 + (1.0 - tau) ** 2 + eps)

                # guidance weight clip: min(β, (1 - τ) / (τ * r_τ^2))  (Alg. 1, line 29; Eq. 2)
                w = min(
                    self.beta, (1.0 - tau) / (max(tau, eps) * r2 + eps)
                )  # WE could also use min(self.beta, tau * inv_r2 / (1 - tau + 1e-8)) to avoid division by zero
                w = torch.tensor(w, device=device).view(1, 1, 1)

                # Euler step (Eq. 1):   A <- A + (1/n) [ vπ + w * g ]
                a_t = a_t + dt * (v_t + w * g)

        # Return A^1
        return a_t

    # ---- Exact soft mask from Eq. (5) ----
    def soft_mask_exact(self, h: int, s: int, d: int, device) -> Tensor:
        """
        Implements Eq. (5) precisely:
          for i in [0, H-1]:
            if i < d:                  W_i = 1
            elif d <= i < H - s:       W_i = c_i * exp(c_i - 1) / (e - 1),
                                       where c_i = (H - s - i) / (H - s - d + 1)
            else:                      W_i = 0
        Broadcast to [1, H, 1] for [B, H, D] tensors.
        """
        weights = torch.zeros(h, device=device, dtype=torch.float32)

        # Region 1: fully frozen prefix (i < d)
        if d > 0:
            weights[:d] = 1.0

        # Region 2: exponential decay over overlap (d <= i < H - s)
        overlap_end = max(0, h - s)
        if d < overlap_end:
            mask = torch.arange(d, overlap_end, device=device, dtype=torch.float32)
            denom = h - s - d + 1
            c = (h - s - mask) / denom
            weights[d:overlap_end] = (
                c * torch.exp(c - 1.0) / (torch.exp(torch.tensor(1.0, device=device)) - 1.0)
            )

        # Region 3: zero in the non-overlap tail (i >= H - s)
        # (already zero by initialization)

        return weights.view(1, h, 1)

    # One denoising call returning vπ(A, o, τ) (velocity field)
    def denoise_step(self, prefix_pad_masks, past_key_values, a_t, tau_tensor):
        """
        Compute vπ(A_t, o, τ) by passing the suffix through the policy head.
        This mirrors Eq. (1–3) where vπ is evaluated at (A_t, o, τ).
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(a_t, tau_tensor)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        with torch.no_grad():
            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)  # vπ
        return v_t

    def reset(self):
        """Reset the state of the VLAFlowMatchingRealTimeCorrected instance."""
        self.current_action = None


class RTCSmolVLAPolicy(SmolVLAPolicy):
    """
    Controller-side adaptor that implements Algorithm 1's GETACTION/INFERENCELOOP
    interface and scheduling (Sec. 4.3):
      • keeps a bounded buffer Q of observed delays (ticks)
      • picks s = max(d, s_min) for the *next* inference start
      • starts the next inference when exactly s actions have been executed
      • swaps to the new chunk as soon as it is available (re-indexing by δ)
    """

    config_class = RTCSmolVLA
    name = "rtc_smolvla"

    def __init__(self, config):
        self.start = False
        super().__init__(config)
        self.start = True

        # Horizons
        self.H = self.config.chunk_size  # prediction horizon
        self.exec_len = self.config.n_action_steps  # number of actions we actually execute per chunk

        # Scheduling hyperparameters (Table 3 in paper gives typical values)
        self.s_min = getattr(
            self.config, "s_min", max(1, min(self.exec_len, getattr(self.config, "inference_steps", 1)))
        )
        self.delay_buffer_size = getattr(self.config, "delay_buffer_size", 10)
        self.initial_delay = getattr(
            self.config, "initial_delay", getattr(self.config, "inference_steps", self.s_min)
        )

        # Delay buffer Q (Alg. 1, line 11); seed with an initial estimate
        self.delay_buffer: deque[int] = deque([int(self.initial_delay)], maxlen=self.delay_buffer_size)

        # RTC state
        self.since_start = 0  # t in Alg. 1 (actions executed since last start)
        self.inference_in_flight = False
        self.s_target = None  # s chosen when we triggered the current inference
        self.needs_init_chunk = True  # first call needs an initial chunk

        # Action queue exposed to the environment
        self._queues[ACTION].clear()

        self.model = self.load_model().to(self.config.device)

        # Start model's background loop
        self.model.start_inference_loop()

    def load_model(self):
        return VLAFlowMatchingRealTimeCorrected(self.config)

    def reset(self):
        super().reset()
        if self.start:
            self.since_start = 0
            self.inference_in_flight = False
            self.s_target = None
            self.needs_init_chunk = True
            self.delay_buffer.clear()
            self.delay_buffer.append(int(self.initial_delay))
            self._queues[ACTION].clear()
            self.model.reset()

    #  Controller entrypoint (Algorithm 1: GETACTION)
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        Called once per control tick Δt by the environment.
        Returns one action and manages RTC scheduling state.

        Matches Alg. 1:
          • lines 3–8: GETACTION increments t and returns A_cur[t-1]
          • lines 13–22: INFERENCELOOP (here: scheduling + background queues)
        """
        self.eval()
        prepared = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, prepared, exclude_keys=[ACTION])

        # 1) If we are mid-rollout, try to "swap as soon as ready" (Alg. 1, lines 19–22).
        self._maybe_swap_to_ready_chunk()

        # 2) If it's time to start the next inference (t == s), trigger it (lines 13–19).
        self._maybe_start_next_inference(prepared)

        # 3) If the action queue ran dry (e.g., very first call), block and fill once.
        if len(self._queues[ACTION]) == 0:
            self._blocking_fill_from_next_chunk(prepared)

        # 4) Pop one action to execute (Alg. 1, line 8) and update counters.
        action = self._queues[ACTION].popleft()
        self.since_start += 1
        return action

    #  Helpers
    @torch.no_grad()
    def _maybe_start_next_inference(self, batch_prepared):
        if self.inference_in_flight:
            return

        # Estimate next delay conservatively as d = max(Q) (Alg. 1, line 17)
        d_est = max(self.delay_buffer) if len(self.delay_buffer) > 0 else self.initial_delay

        # Choose execution horizon s = max(d, s_min) for this round.
        s = int(min(self.exec_len - 1, max(d_est, self.s_min)))  # clamp so there is at least one step after s
        # Start when we have executed exactly s actions since the last start (t == s).
        if self.since_start == s or self.needs_init_chunk:
            obs_rt = self.get_observation_realtime(batch_prepared)
            # Package real-time scheduling info for the model thread.
            self.model.input_queue.put((*obs_rt, s, d_est))
            self.s_target = s
            self.inference_in_flight = True
            self.needs_init_chunk = False

    @torch.no_grad()
    def _maybe_swap_to_ready_chunk(self):
        if not self.inference_in_flight:
            return
        try:
            # Non-blocking: swap as soon as new chunk is ready (Alg. 1, line 20)
            action_new = self.model.output_queue.get_nowait()  # [B, H, D]
        except Empty:
            return

        # Observed delay δ in *ticks since we triggered*, i.e., δ = t - s   (Alg. 1, line 22)
        delta = max(0, self.since_start - int(self.s_target or 0))
        self.delay_buffer.append(int(delta))

        # Postprocess to env action space and *re-index* so the next action is A_new[δ]
        actions = self._postprocess_chunk(action_new)  # [B, H, action_dim]
        actions_transpose = actions.transpose(0, 1)  # [H, B, action_dim]
        tail = actions_transpose[delta : delta + self.exec_len]  # next actions to enqueue
        self._queues[ACTION].clear()
        self._queues[ACTION].extend(tail)

        # Reset t so that it now indexes into A_new: t <- t - s = δ  (Alg. 1, line 21)
        self.since_start = int(delta)
        self.inference_in_flight = False
        self.s_target = None

    @torch.no_grad()
    def _blocking_fill_from_next_chunk(self, batch_prepared):
        """
        Used for the very first fill (no actions available yet).
        Blocks until the initial chunk arrives and fills the action queue.
        """
        # If no inference is in flight, start one with s=0.
        if not self.inference_in_flight:
            obs_rt = self.get_observation_realtime(batch_prepared)
            self.model.input_queue.put((*obs_rt, 0, max(self.delay_buffer)))
            self.s_target = 0
            self.inference_in_flight = True
            self.needs_init_chunk = False

        # Block until a chunk is ready
        action_new = self.model.output_queue.get()
        self.delay_buffer.append(0)  # δ = 0 at initialization
        actions = self._postprocess_chunk(action_new)
        actions_transpose = actions.transpose(0, 1)
        self._queues[ACTION].extend(actions_transpose[: self.exec_len])
        self.since_start = 0
        self.inference_in_flight = False
        self.s_target = None

    @torch.no_grad()
    def _postprocess_chunk(self, actions: Tensor) -> Tensor:
        """
        Map model outputs to environment action space.
        - Remove any padding dims added for the model head.
        - Optional π-ALOHA encoding if enabled.
        """
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    @torch.no_grad()
    def get_observation_realtime(self, batch):
        """
        Package the inputs (images, language, state), stacking any available queues.
        Matches your original helper; used to build the "prefix" in the model thread.
        """
        for k in batch:
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        return images, img_masks, lang_tokens, lang_masks, state
