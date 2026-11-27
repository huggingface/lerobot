# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/deploy/async_exec/memory_rtc.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import math

import torch

from lerobot.policies.sana_robot.sana_robot_core.ar_session import ARSession
from lerobot.policies.sana_robot.sana_robot_core.view_stream import ViewStream


class PhaseMemory:
    """One memory of SANA-Robot-AR per phase, for chunks generated every P control steps.

    In training a chunk that starts at step t reads the sink and the windows of C steps that end at t, t - C,
    t - 2C and so on, with copies of the first frame before the start of the episode. A chunk generated every P
    steps starts at steps of C / gcd(C, P) phases modulo C, and the windows one phase reads are never another's.
    So there is one memory per phase: the session's memory and the state of the video stream. Every gcd(C, P)
    steps the memory of that step's phase is loaded, the window that ends there is encoded and committed, and the
    memory is saved back. A chunk is generated on the memory of its step's phase.
    """

    def __init__(
        self, session: ARSession, stream: ViewStream, n_action_steps: int, frame_stride: int
    ) -> None:
        """Build the memories.

        Args:
            session: The session.
            stream: The video stream.
            n_action_steps: P, the control steps between two generated chunks: a multiple of ``frame_stride``, at
                most a chunk.
            frame_stride: The control steps between two video frames.
        """
        self.session = session
        self.stream = stream
        self.chunk_size = session.chunk_actions
        self.n_action_steps = int(n_action_steps)
        self.frame_stride = int(frame_stride)
        self.window_steps = math.gcd(self.chunk_size, self.n_action_steps)
        self.phases = self.chunk_size // self.window_steps
        self.memories = []
        self.active = None
        self.sink = None
        self.observation = None

    def phase(self, step: int) -> int:
        """Return the phase of a control step."""
        return step % self.chunk_size // self.window_steps

    def open(self, first_frames) -> torch.Tensor:
        """Open an episode at its first frame; every phase's memory starts from the sink. Returns the sink latent."""
        self.sink = self.stream.start(first_frames)
        self.session.start(self.sink)
        opened = (self.session.save_memory(), self.stream.save())
        self.memories = [opened] * self.phases
        self.active = 0
        return self.sink

    def commit_window(self, step: int, window) -> torch.Tensor:
        """Commit the window that ends at ``step`` to the memory of its phase.

        Args:
            step: A positive multiple of gcd(C, P).
            window: Per camera, the last C / frame_stride video frames up to the one at ``step``, [3, T, h, w];
                copies of the first frame stand in for frames before the episode.

        Returns:
            The latents of the window [1, C, f, 1, tokens].
        """
        phase = self.phase(step)
        self._switch(phase)
        latents = self.stream.encode_chunk(window)
        self.session.commit(latents)
        self.memories[phase] = (self.session.save_memory(), self.stream.save())
        if step % self.n_action_steps == 0:
            self.observation = self.stream.encode_observation([clip[:, -1:] for clip in window])
        return latents

    def generate(self, step: int, state, state_mask, action_mask, **noise):
        """Generate the chunk that starts at ``step`` on the memory of its phase; see ``ARSession.generate``."""
        self._switch(self.phase(step))
        observation = self.sink if step == 0 else self.observation
        return self.session.generate(observation, state, state_mask, action_mask, **noise)

    def _switch(self, phase: int) -> None:
        """Load the memory of a phase into the session and the stream."""
        if phase != self.active:
            (memory, stream_state) = self.memories[phase]
            self.session.load_memory(memory)
            self.stream.load(stream_state)
            self.active = phase


class MemoryRTCController:
    """Memory RTC: asynchronous execution of SANA-Robot-AR on one memory per phase.

    A chunk is generated every P control steps from the state and the frames observed at its first step, and takes
    over d steps later, 0 <= d <= P <= C - d; its first d steps, which the robot ran from the chunk before while
    it was generated, are dropped. P = C - d by default. With d = 0 the robot waits for every chunk: P = C is
    synchronous execution on a single memory, P < C replans inside a chunk.
    """

    def __init__(
        self,
        session: ARSession,
        stream: ViewStream,
        async_delay_steps: int,
        n_action_steps: int | None = None,
    ) -> None:
        """Build the controller.

        Args:
            session: The session.
            stream: The video stream.
            async_delay_steps: d, the control steps the robot runs from the running chunk while the next one is
                generated; a multiple of the video frame stride.
            n_action_steps: P, the control steps between two generated chunks; C - d when None.
        """
        self.chunk_size = session.chunk_actions
        self.async_delay_steps = int(async_delay_steps)
        self.n_action_steps = (
            self.chunk_size - self.async_delay_steps if n_action_steps is None else int(n_action_steps)
        )
        self.frame_stride = session.policy.video_frame_stride(session.data_info)
        self.memory = PhaseMemory(session, stream, self.n_action_steps, self.frame_stride)
        self.window_steps = self.memory.window_steps

    def open(self, window, state, state_mask, action_mask, **noise) -> torch.Tensor:
        """Open an episode at its first frame and generate chunk 0. Returns its normalized actions [1, C, 80]."""
        self.memory.open([clip[:, :1] for clip in window])
        return self.memory.generate(0, state, state_mask, action_mask, **noise)[0]

    def observe(self, step: int, window) -> None:
        """Commit the window that ends at a step where no chunk is generated."""
        self.memory.commit_window(step, window)

    def generate(
        self, step: int, window, state, state_mask, action_mask, running=None, **noise
    ) -> torch.Tensor:
        """Commit the window that ends at ``step`` and generate the chunk that starts there. Returns its normalized actions [1, C, 80]."""
        self.memory.commit_window(step, window)
        return self.memory.generate(step, state, state_mask, action_mask, **noise)[0]
