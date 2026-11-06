#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Diffusion sampling that torch.export can trace, shared by diffusion and multi_task_dit."""

from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _diffusers_available

from .export_utils import is_non_strict_exporting

if TYPE_CHECKING or _diffusers_available:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    class ExportableDDPMScheduler(DDPMScheduler):
        """A ``DDPMScheduler`` that ``torch.export`` can trace when stepped with Python-int timesteps.

        diffusers finds the previous timestep with a tensor search and branches on the result, which export
        cannot trace. This does the same lookup on a Python list. Before ``set_timesteps`` the timesteps count
        down by one, so the lookup also gives diffusers' ``timestep - 1`` fallback.
        """

        def previous_timestep(self, timestep: int) -> int:
            timesteps = self.timesteps.tolist() + [-1]
            return timesteps[timesteps.index(timestep) + 1]

else:
    DDIMScheduler = None
    ExportableDDPMScheduler = None


class InferenceStepsMixin:
    """``num_inference_steps`` for a module that samples with its ``noise_scheduler``.

    Sampling loops over the scheduler's timesteps as Python ints. Non-strict torch.export cannot run
    ``set_timesteps`` while tracing, because the timesteps it makes would be symbolic, so assigning
    ``num_inference_steps`` sets them. Eager and compiled calls set them again on every call, so a replaced
    scheduler samples the configured steps. An export trace instead checks that the scheduler still holds the
    schedule set last.
    """

    noise_scheduler: "DDPMScheduler | DDIMScheduler"

    @property
    def num_inference_steps(self) -> int:
        return self._num_inference_steps

    @num_inference_steps.setter
    def num_inference_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"num_inference_steps must be at least 1, got {value}")
        self._set_schedule(value)

    def _set_schedule(self, num_inference_steps: int) -> None:
        self.noise_scheduler.set_timesteps(num_inference_steps)
        self._num_inference_steps = num_inference_steps
        self._schedule = _schedule_of(self.noise_scheduler)

    def inference_timesteps(self) -> list[int]:
        """The timesteps to sample, as Python ints. Eager and compiled calls set them first; export checks them."""
        if not is_non_strict_exporting():
            # Not through the setter: under torch.compile, this frame would not see what a setter changed.
            self._set_schedule(self._num_inference_steps)
        elif not isinstance(self.noise_scheduler, (DDIMScheduler, ExportableDDPMScheduler)):
            raise RuntimeError(
                "torch.export can trace sampling only with DDIMScheduler or lerobot's "
                f"ExportableDDPMScheduler, not {type(self.noise_scheduler).__name__}."
            )
        elif _schedule_of(self.noise_scheduler) != self._schedule:
            raise RuntimeError(
                "The noise scheduler no longer holds the schedule set by num_inference_steps, for example "
                "after it was replaced or reconfigured. Assign num_inference_steps on the "
                f"{type(self).__name__}, not on its noise scheduler, or run one eager sample, before tracing."
            )
        return self.noise_scheduler.timesteps.tolist()


def _schedule_of(scheduler: Any) -> tuple[dict[str, Any], int, list[int]]:
    return dict(scheduler.config), scheduler.num_inference_steps, scheduler.timesteps.tolist()
