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

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel  # noqa: E402
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig  # noqa: E402
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (  # noqa: E402
    DiffusionObjective,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_STATE,
)


class Sampler(torch.nn.Module):
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, global_cond, noise):
        return self.model.conditional_sample(noise.shape[0], global_cond=global_cond, noise=noise)


class Denoiser(torch.nn.Module):  # a stand-in for the DiT
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, sample, timestep, conditioning_vec):
        return self.proj(sample) * timestep[:, None, None] / 100 + conditioning_vec[:, None]


class ObjectiveSampler(torch.nn.Module):
    def __init__(self, objective: DiffusionObjective, denoiser: Denoiser):
        super().__init__()
        self.objective, self.denoiser = objective, denoiser

    def forward(self, conditioning_vec, noise):
        return self.objective.conditional_sample(self.denoiser, noise.shape[0], conditioning_vec, noise=noise)


def make_sampler(kind, scheduler, num_inference_steps=10):
    """A diffusion model or Multi-Task DiT objective, a module sampling with it, and its inputs."""
    features = {
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        "noise_scheduler_type": scheduler,
        "num_train_timesteps": 100,
        "num_inference_steps": num_inference_steps,
    }
    if kind == "diffusion":
        config = DiffusionConfig(
            horizon=8,
            n_action_steps=4,
            down_dims=(16, 32),
            diffusion_step_embed_dim=16,
            n_groups=4,
            **features,
        )
        model = DiffusionModel(config).eval()
        return model, Sampler(model), (torch.randn(2, 4 * config.n_obs_steps), torch.randn(2, 8, 2))
    config = MultiTaskDiTConfig(objective="diffusion", **features)
    objective = DiffusionObjective(config, action_dim=2, horizon=8)
    return objective, ObjectiveSampler(objective, Denoiser()), (torch.randn(2, 2), torch.randn(2, 8, 2))


def sample(sampler, args):
    torch.manual_seed(0)
    return sampler(*args)


@pytest.mark.parametrize("kind", ["diffusion", "multi_task_dit"])
def test_exported_sampling_matches_eager(kind):
    with pytest.raises(ValueError, match="at least 1"):
        make_sampler(kind, "DDPM", num_inference_steps=0)
    with pytest.raises(ValueError, match="cannot be larger"):
        make_sampler(kind, "DDPM", num_inference_steps=1000)
    model, sampler, args = make_sampler(kind, "DDPM")
    # Exported right after build, as after loading a policy, so it relies on the schedule the constructor
    # set. Strict tracing cannot read the timesteps as ints.
    exported = torch.export.export(sampler, args, strict=False).module()
    assert torch.equal(sample(exported, args), sample(sampler, args))

    model.num_inference_steps = 5
    assert len(model.noise_scheduler.timesteps) == 5
    with pytest.raises(ValueError, match="at least 1"):
        model.num_inference_steps = 0
    with pytest.raises(ValueError, match="cannot be larger"):
        model.num_inference_steps = 1000
    assert model.num_inference_steps == 5
    # Exported right after the assignment, so it relies on the schedule the assignment set.
    exported = torch.export.export(sampler, args, strict=False).module()
    assert torch.equal(sample(exported, args), sample(sampler, args))


@pytest.mark.parametrize("kind", ["diffusion", "multi_task_dit"])
def test_tracing_refuses_a_schedule_that_eager_sampling_would_reset(kind):
    def export(sampler, args):
        return torch.export.export(sampler, args, strict=False).module()

    # A changed DDIM spacing keeps 10 steps. torch.compile resets the schedule like eager sampling.
    model, sampler, args = make_sampler(kind, "DDIM")
    model.noise_scheduler.register_to_config(timestep_spacing="trailing")
    stale = f"Assign num_inference_steps on the {type(model).__name__}"
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)
    assert torch.equal(sample(torch.compile(sampler, backend="eager"), args), sample(sampler, args))
    assert torch.equal(sample(export(sampler, args), args), sample(sampler, args))

    # A DDIM step reads the scheduler's own step count, so setting only that count makes the schedule stale.
    model, sampler, args = make_sampler(kind, "DDIM")
    model.noise_scheduler.num_inference_steps = 20
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)

    # Custom DDPM timesteps keep the step count at 10. Assigning it on the model sets the schedule again.
    model, sampler, args = make_sampler(kind, "DDPM")
    model.noise_scheduler.set_timesteps(timesteps=[99, 49, 0])
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)
    model.num_inference_steps = 10
    assert torch.equal(sample(export(sampler, args), args), sample(sampler, args))

    # A swapped scheduler whose own step count is set to 10 still holds its default timesteps.
    model.noise_scheduler = diffusers.DDIMScheduler.from_config(model.noise_scheduler.config)
    model.noise_scheduler.num_inference_steps = 10
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)
    model.noise_scheduler = diffusers.DDPMScheduler.from_config(model.noise_scheduler.config)
    model.num_inference_steps = 10
    with pytest.raises(RuntimeError, match="only with DDIMScheduler or lerobot's ExportableDDPMScheduler"):
        export(sampler, args)
