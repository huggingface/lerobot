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

"""A DDPM scheduler that torch.export can trace, shared by diffusion and multi_task_dit."""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _diffusers_available

if TYPE_CHECKING or _diffusers_available:
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
    ExportableDDPMScheduler = None
