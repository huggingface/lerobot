# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/deploy/memory/softmax_memory.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch

from lerobot.policies.sana_robot.sana_robot_core.teacher_forcing import SINK_HORIZON_CHUNKS

SINK_READ = "sink"
COMMIT_READ = "commit"
OBSERVATION_READ = "observation"
DENOISE_READ = "denoise"
MIN_CLOCK_CHUNKS = 64


class SoftmaxMemory:
    """What the softmax layers of SANA-Robot-AR read in a streaming window, shared by the layers of one session.

    A read holds, in this order, the sink (the episode's first frame), the last ``window_chunks`` committed chunks,
    the observation of the chunk being generated (its latent and its state row) and the window itself: the chunk a
    commit writes, the observation, or the noisy targets and action rows of a denoising step. Every token sits at
    its instant on the episode clock: a target latent at the middle of the frames it covers, an observation latent
    and the state row at the start of their chunk, action row a one step after it, and the sink at the start of
    the chunk ``SINK_HORIZON_CHUNKS`` before the reader's.
    """

    def __init__(
        self, view_areas, chunk_frames: int, chunk_actions: int, window_chunks: int = 1, vae_stride: int = 8
    ):
        self.view_areas = tuple(int(area) for area in view_areas)
        self.chunk_frames = int(chunk_frames)
        self.chunk_actions = int(chunk_actions)
        self.window_chunks = int(window_chunks)
        self.time_units = 2 * int(vae_stride)
        self.target_offset = int(vae_stride) + 1
        self.frame_tokens = sum(self.view_areas)
        self.entry_tokens = self.chunk_frames * self.frame_tokens
        self.observation_tokens = self.frame_tokens + 1
        self.capacity = (
            self.frame_tokens
            + self.window_chunks * self.entry_tokens
            + self.observation_tokens
            + self.entry_tokens
            + self.chunk_actions
        )
        self.layers = []
        self.count = 0
        self.sink_written = False
        self.clock_floor = MIN_CLOCK_CHUNKS
        self.rope_tables = None
        self._phases = {}
        self._phase_states = {}

    def add_layer(self) -> SoftmaxLayerMemory:
        """Return the store of one more softmax layer of the session."""
        layer = SoftmaxLayerMemory(self)
        self.layers.append(layer)
        return layer

    @property
    def kept(self) -> int:
        """Return how many committed chunks a read holds."""
        return min(self.count, self.window_chunks)

    @property
    def static_tokens(self) -> int:
        """Return the tokens of the sink and the kept chunks, which only a commit changes."""
        return self.frame_tokens + self.kept * self.entry_tokens if self.sink_written else 0

    @property
    def clock_chunks(self) -> int:
        """Return how many chunks the rotary tables must cover: the episode so far rounded up to a power of two, and at least ``clock_floor``."""
        return max(self.clock_floor, 1 << self.count.bit_length())

    def read_kind(self) -> str:
        """Return which read a commit is: the sink's own before the sink is written, a chunk's after it."""
        return COMMIT_READ if self.sink_written else SINK_READ

    def window_offset(self, read: str) -> int:
        """Return the token at which the window of a read starts: after the observation for a denoising step, right after the kept chunks otherwise."""
        return self.static_tokens + (self.observation_tokens if read == DENOISE_READ else 0)

    def advance(self) -> None:
        """Make the window the last forward committed part of the memory: the sink, or one more chunk in place of the oldest kept one."""
        if not self.sink_written:
            self.sink_written = True
            return
        if self.kept == self.window_chunks:
            (first, last) = (self.frame_tokens, self.static_tokens)
            for layer in self.layers:
                layer.shift(first, last, self.entry_tokens)
        self.count += 1

    def reset(self) -> None:
        """Forget the episode."""
        self.count = 0
        self.sink_written = False

    def save(self) -> dict:
        """Return a copy of the memory for ``load``."""
        layers = [layer.save() for layer in self.layers]
        return {"count": self.count, "sink_written": self.sink_written, "layers": layers}

    def load(self, saved: dict) -> None:
        """Bring a saved memory back into the stores, which keep their addresses."""
        self.count = saved["count"]
        self.sink_written = saved["sink_written"]
        for layer, stored in zip(self.layers, saved["layers"], strict=False):
            layer.load(stored)

    def phases(self, read: str) -> torch.Tensor:
        """Return the rotary phases [1, 1, T, head_dim / 2] of every key of a read. A read of a given length keeps one buffer, which a captured CUDA graph reads at its address."""
        tables = self.rope_tables
        state = (self.count, self.sink_written, id(tables))
        current = self._phase_states.get(read)
        if current is not None and current[0] == state:
            return current[1]
        rows = self._phase_rows(read, tables)
        key = (read, rows.shape[2])
        buffer = self._phases.get(key)
        if buffer is not None and buffer.device == rows.device and buffer.dtype == rows.dtype:
            buffer.copy_(rows)
        else:
            self._phases[key] = buffer = rows
        self._phase_states[read] = (state, buffer)
        return buffer

    def refresh_phases(self) -> None:
        """Bring the phases of every read up to date with the memory, outside any captured graph."""
        if self.rope_tables is None:
            return
        reads = (COMMIT_READ, OBSERVATION_READ, DENOISE_READ) if self.sink_written else (SINK_READ,)
        for read in reads:
            self.phases(read)

    def _phase_rows(self, read: str, tables: dict) -> torch.Tensor:
        """Assemble the phases of a read from the policy's tables: a video token takes its time row next to its spatial row, a robot row its step's row."""
        (frames, actions, chunk) = (self.chunk_frames, self.chunk_actions, self.count)
        device = tables["time"].device
        rows = []
        if read == SINK_READ:
            rows.append(self._video_rows(tables, 0, 0, 1))
        else:
            sink_chunk = max(0, chunk - SINK_HORIZON_CHUNKS)
            rows.append(self._video_rows(tables, sink_chunk * frames, 0, 1))
            for entry in range(chunk - self.kept, chunk):
                rows.append(self._video_rows(tables, entry * frames, self.target_offset, frames))
            if read != COMMIT_READ:
                rows.append(self._video_rows(tables, chunk * frames, 0, 1))
                rows.append(tables["robot"][chunk * actions : chunk * actions + 1])
            if read != OBSERVATION_READ:
                rows.append(self._video_rows(tables, chunk * frames, self.target_offset, frames))
            if read == DENOISE_READ:
                steps = chunk * actions + 1 + torch.arange(actions, device=device)
                rows.append(tables["robot"][steps])
        return torch.cat(rows)[None, None]

    def _video_rows(self, tables: dict, first_frame: int, offset: int, frames: int) -> torch.Tensor:
        """Return the phases of ``frames`` latent frames of every view from latent frame ``first_frame``, in the stored order: view by view, frame by frame."""
        device = tables["time"].device
        time = (first_frame + torch.arange(frames, device=device)) * self.time_units + offset
        rows = []
        first_token = 0
        for area in self.view_areas:
            tokens = first_token + torch.arange(area, device=device)
            time_rows = tables["time"][time.repeat_interleave(area)]
            rows.append(torch.cat((time_rows, tables["spatial"][tokens.repeat(frames)]), dim=-1))
            first_token += area
        return torch.cat(rows)


class SoftmaxLayerMemory:
    """The keys and values of one softmax layer before rotation, in read order; the object the policy finds under ``"kv"`` in the layer's cache."""

    def __init__(self, memory: SoftmaxMemory):
        self.memory = memory
        self.keys = None
        self.values = None

    @property
    def clock_chunks(self) -> int:
        return self.memory.clock_chunks

    @property
    def rope_tables(self):
        return self.memory.rope_tables

    @rope_tables.setter
    def rope_tables(self, tables) -> None:
        self.memory.rope_tables = tables

    def commit(self, keys, values, queries=None):
        """Write the sink or a clean chunk after the kept chunks and return its read: (keys, values, None, phases)."""
        return self._read(keys, values, self.memory.read_kind())

    def observe(self, keys, values, queries=None):
        """Write the observation of the chunk being generated after the kept chunks and return its read."""
        return self._read(keys, values, OBSERVATION_READ)

    def read(self, keys, values, queries=None):
        """Write the window of a denoising step after the observation and return its read."""
        return self._read(keys, values, DENOISE_READ)

    def shift(self, first: int, last: int, tokens: int) -> None:
        """Drop the oldest kept chunk: move the tokens from ``first + tokens`` to ``last + tokens`` back by one chunk."""
        self.keys[:, first:last] = self.keys[:, first + tokens : last + tokens].clone()
        self.values[:, first:last] = self.values[:, first + tokens : last + tokens].clone()

    def save(self):
        """Return a copy of the store, or None before its first write."""
        return None if self.keys is None else (self.keys.clone(), self.values.clone())

    def load(self, stored) -> None:
        """Copy a saved store into this one's buffers."""
        if stored is not None:
            self._allocate(stored[0])
            self.keys.copy_(stored[0])
            self.values.copy_(stored[1])

    def _allocate(self, like: torch.Tensor) -> None:
        """Create the buffers on the first write, for every token a read can hold."""
        if self.keys is None:
            shape = (like.shape[0], self.memory.capacity, *like.shape[2:])
            self.keys = like.new_zeros(shape)
            self.values = like.new_zeros(shape)

    def _read(self, keys, values, read: str):
        self._allocate(keys)
        start = self.memory.window_offset(read)
        end = start + keys.shape[1]
        self.keys[:, start:end] = keys
        self.values[:, start:end] = values
        return (self.keys[:, :end], self.values[:, :end], None, self.memory.phases(read))
