#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist

from .utils import format_big_number

_VALID_REDUCTIONS = ("none", "max", "mean", "sum")


class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py

    Args:
        name: Display name of the metric.
        fmt: Format string used when rendering the metric.
        reduction: Cross-process reduction applied by :meth:`MetricsTracker.reduce_across_ranks`
            before logging. One of ``"none"`` (per-rank value, default), ``"max"``, ``"mean"``,
            or ``"sum"``. Use ``"max"`` for bottleneck-style metrics (e.g. dataloading or
            update wall time) so multi-GPU runs report the slowest rank rather than rank 0.
    """

    def __init__(self, name: str, fmt: str = ":f", reduction: str = "none"):
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(
                f"Invalid reduction {reduction!r} for AverageMeter; expected one of {_VALID_REDUCTIONS}."
            )
        self.name = name
        self.fmt = fmt
        self.reduction = reduction
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """
    A helper class to track and log metrics over time.

    Args:
        batch_size (int): Per-process batch size (samples per micro-batch on each
            data-parallel worker).
        num_frames (int): Total number of frames in the training dataset.
        num_episodes (int): Total number of episodes in the training dataset.
        metrics (dict[str, AverageMeter]): The meters to track, keyed by metric name.
        initial_step (int): Step counter to start from (non-zero when resuming a run).
            Defaults to 0.
        dp_world_size (int): Number of distinct data-parallel workers
            (`dp_replicate * dp_shard`), used to scale sample accounting; context-parallel
            peers consume the same batch and must not be double counted. Defaults to 1.

    Usage pattern:

    ```python
    # initialize, potentially with non-zero initial step (e.g. if resuming run)
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_metrics = MetricsTracker(
        batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        metrics,
        initial_step=step,
        dp_world_size=dp_world,
    )

    # update metrics derived from step (samples, episodes, epochs) at each training step
    train_metrics.step()

    # update various metrics
    loss = policy.forward(batch)
    train_metrics.loss = loss

    # display current metrics
    logging.info(train_metrics)

    # export for wandb
    wandb.log(train_metrics.to_dict())

    # reset averages after logging
    train_metrics.reset_averages()
    ```
    """

    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "_dp_world_size",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
        dp_world_size: int = 1,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        # Sample accounting scales by the number of DISTINCT data-parallel workers, which is
        # dp_replicate * dp_shard — not the world size: context-parallel peers consume the same
        # batch and must not be double counted. `step` counts micro-batches, so no
        # grad-accumulation factor belongs here either.
        self._dp_world_size = dp_world_size
        self.metrics = metrics

        self.steps = initial_step
        # A sample is an (observation,action) pair, where observation and action
        # can be on multiple timestamps. In a batch, we have `batch_size` number of samples.
        self.samples = self.steps * self._batch_size * self._dp_world_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """
        Updates metrics that depend on 'step' for one step.
        """
        self.steps += 1
        self.samples += self._batch_size * self._dp_world_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def reduce_across_ranks(self) -> None:
        """
        Synchronises the running averages of every metric whose ``reduction`` is not ``"none"``
        across all distributed processes (in-place).

        This is a collective operation and MUST be invoked on every rank — typically just before
        logging. Outside distributed runs it is a no-op. Without it, metrics reported by the
        main process only reflect rank 0; for bottleneck-style timings (``dataloading_s``,
        ``update_s``, ...) that means the slowest worker's stall is invisible.

        Torch-native on purpose: metrics code carries no Accelerator dependency.
        Note the reduction spans the WORLD group — correct for count-free averages (loss values
        are identical within a context-parallel group, so including CP peers is a weighted
        no-op).
        """
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return

        buckets: dict[str, list[str]] = defaultdict(list)
        for name, meter in self.metrics.items():
            if meter.reduction != "none":
                buckets[meter.reduction].append(name)
        if not buckets:
            return

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        reduce_ops = {
            "mean": dist.ReduceOp.AVG,
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
        }
        for reduction, names in buckets.items():
            tensor = torch.tensor([self.metrics[n].avg for n in names], dtype=torch.float32, device=device)
            dist.all_reduce(tensor, op=reduce_ops[reduction])
            for name, value in zip(names, tensor.tolist(), strict=True):
                meter = self.metrics[name]
                # Preserve avg == sum / count so a later .update() on this meter accumulates
                # against the cluster view, not the stale per-rank history.
                meter.avg = value
                meter.sum = value * meter.count

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        Returns the current metric values (or averages if `use_avg=True`) as a dict.
        """
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()
