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

"""Top-level pipeline config for distributed RL training (actor / learner)."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig

from .algorithms.configs import RLAlgorithmConfig
from .algorithms.factory import make_algorithm_config
from .algorithms.sac import SACAlgorithmConfig  # noqa: F401


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    """Top-level config for the actor/learner distributed RL training server.

    Extends [`~configs.train.TrainPipelineConfig`] with an optional (rather than required) offline
    `dataset` and the RL-specific algorithm/data-mixing fields below.

    Args:
        env (`lerobot.envs.configs.EnvConfig | None`, *optional*):
            Simulation environment configuration, used for `env_eval_freq` evaluation rollouts.
        policy (`lerobot.configs.policies.PreTrainedConfig | None`, *optional*):
            The actor policy configuration.
        reward_model (`lerobot.configs.rewards.RewardModelConfig | None`, *optional*):
            Reward model configuration, when training a reward model instead of a policy.
        output_dir (`pathlib.Path | None`, *optional*):
            Directory to save run outputs to. Reusing the same value across runs overwrites its
            contents unless `resume` is `True`.
        job_name (`str | None`, *optional*):
            Name used for logging and checkpoint directory naming.
        resume (`bool`, *optional*, defaults to `False`):
            Whether to resume a previous run from `--config_path`'s checkpoint.
        seed (`int | None`, *optional*, defaults to 1000):
            Random seed for model initialization, dataset shuffling, and evaluation environments.
        cudnn_deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic cuDNN algorithms for reproducibility. Disables
            `cudnn.benchmark`, which may reduce training speed.
        num_workers (`int`, *optional*, defaults to 4):
            Number of dataloader worker processes.
        batch_size (`int`, *optional*, defaults to 8):
            Offline-dataset dataloader batch size.
        prefetch_factor (`int`, *optional*, defaults to 4):
            Number of batches prefetched per dataloader worker.
        persistent_workers (`bool`, *optional*, defaults to `True`):
            Whether dataloader workers stay alive between epochs.
        dataloader_multiprocessing_context (`str | None`, *optional*, defaults to `"spawn"`):
            DataLoader worker start method. `None` uses Python's platform default.
        steps (`int`, *optional*, defaults to 100000):
            Total number of training steps.
        env_eval_freq (`int`, *optional*, defaults to 20000):
            Run the policy in the simulation environment every N steps to measure reward/success.
            `0` disables environment evaluation.
        log_freq (`int`, *optional*, defaults to 200):
            Log training metrics every N steps.
        eval_steps (`int`, *optional*, defaults to 0):
            Compute eval loss on held-out episodes every N steps. `0` disables it.
        max_eval_samples (`int`, *optional*, defaults to 0):
            Cap on total eval samples, split uniformly across tasks. `0` uses all held-out data.
        tolerance_s (`float`, *optional*, defaults to 0.0001):
            Maximum timestamp tolerance, in seconds, when loading dataset frames.
        save_checkpoint (`bool`, *optional*, defaults to `True`):
            Whether to save training checkpoints at all.
        save_freq (`int`, *optional*, defaults to 20000):
            Save a checkpoint every N training steps, and after the last step. A non-positive value
            disables periodic saving, keeping only the final checkpoint.
        checkpoint_format (`CheckpointFormat`, *optional*, defaults to `CheckpointFormat.SAFETENSORS`):
            Model-artifact format inside checkpoints.
        use_policy_training_preset (`bool`, *optional*, defaults to `True`):
            Whether to use the policy's own recommended optimizer/scheduler preset when `optimizer`/
            `scheduler` are unset.
        optimizer (`lerobot.optim.optimizers.OptimizerConfig | None`, *optional*):
            Optimizer configuration override.
        scheduler (`lerobot.optim.schedulers.LRSchedulerConfig | None`, *optional*):
            Learning-rate scheduler configuration override.
        parallelism (`ParallelismConfig`, *optional*):
            Process topology: `dp_replicate`/`dp_shard` (HSDP) and context-parallel degree.
        accelerator (`AcceleratorConfig`, *optional*):
            Execution runtime handed to the Accelerator: mixed precision, gradient accumulation,
            FSDP/DDP tuning knobs, compile & activation-checkpointing.
        eval (`EvalConfig`, *optional*):
            Simulation-environment evaluation configuration (number of episodes, batch size).
        wandb (`WandBConfig`, *optional*):
            Weights & Biases logging configuration.
        peft (`lerobot.configs.default.PeftConfig | None`, *optional*):
            PEFT (e.g. LoRA) adapter configuration for parameter-efficient fine-tuning.
        job (`JobConfig`, *optional*):
            Where to run training: local (default) or an HF Jobs flavor.
        save_checkpoint_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push each saved checkpoint to the Hub as it is written, not just the final
            model.
        sample_weighting (`lerobot.utils.sample_weighting.SampleWeightingConfig | None`, *optional*):
            Sample weighting configuration (e.g. for RA-BC training).
        rename_map (`dict`, *optional*):
            Mapping to override observation image/state key names.
        dataset (`DatasetConfig | None`, *optional*):
            Optional offline dataset config. Unlike imitation-learning training, RL doesn't require an
            offline dataset — data comes from the online replay buffer.
        algorithm (`RLAlgorithmConfig | None`, *optional*):
            RL algorithm configuration. Defaults to a SAC config (with `policy_config` populated from
            `self.policy`) in `validate` when unset.
        mixer (`str`, *optional*, defaults to `"online_offline"`):
            Data mixer strategy name. Currently only `"online_offline"` is supported.
        online_ratio (`float`, *optional*, defaults to 0.5):
            Fraction of each training batch sampled from the online replay buffer when using
            `OnlineOfflineMixer`; the remainder comes from the offline dataset.
    """

    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional

    # Algorithm config.
    algorithm: RLAlgorithmConfig | None = None

    # Data mixer strategy name. Currently supports "online_offline".
    mixer: str = "online_offline"
    # Fraction sampled from online replay when using OnlineOfflineMixer.
    online_ratio: float = 0.5

    def validate(self) -> None:
        """See [`~configs.train.TrainPipelineConfig.validate`].

        Additionally defaults `algorithm` to a SAC config and populates its `policy_config` from
        `self.policy` when unset.
        """
        super().validate()

        if self.algorithm is None:
            self.algorithm = make_algorithm_config("sac")

        if getattr(self.algorithm, "policy_config", None) is None:
            self.algorithm.policy_config = self.policy
