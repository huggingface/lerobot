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
"""CPU/Gloo parity of actual weighted optimizer updates and their batch reference."""

import os
import socket
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pd = pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from accelerate import Accelerator  # noqa: E402

from lerobot.scripts.lerobot_train import update_policy  # noqa: E402
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # noqa: E402
from lerobot.utils.sample_weighting import SampleWeightingConfig, make_sample_weighter  # noqa: E402

pytestmark = pytest.mark.dataset


class _ScalarPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(chunk_size=1)

    def forward(self, batch: dict, reduction: str = "mean") -> tuple[torch.Tensor, dict]:
        loss = (self.weight - batch["target"]).square()
        return (loss if reduction == "none" else loss.mean()), {}


def _weighted_worker(rank: int, port: int, directory: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        ACCELERATE_USE_CPU="true",
    )
    torch.set_num_threads(1)
    root = Path(directory)
    # Cases vary data/assignment, not the policy, learning rate, or optimizer state.
    for name, deltas, order, epsilon, fallback in [
        ("mixed", [0.2, 0.2, 0.2, -0.2], [0, 1, 2, 3], 1e-6, 1.0),
        ("permuted", [0.2, 0.2, 0.2, -0.2], [0, 2, 1, 3], 1e-6, 1.0),
        ("soft", [0.001, 0.002, 0.009, 0.009], [0, 1, 2, 3], 1e-6, 1.0),
        ("empty_rank", [0.2, 0.2, -0.2, -0.2], [0, 1, 2, 3], 1e-6, 1.0),
        ("all_zero", [-0.2, -0.2, -0.2, -0.2], [0, 1, 2, 3], 1e-6, 1.0),
        ("tiny", [float("nan")] * 4, [0, 1, 2, 3], 0.1, 1e-10),
        ("missing_index", [0.2] * 4, [0, 1, 2, 3], 1.0, 1.0),
        ("mixed_index", [0.2, -0.2, 0.2, -0.2], [0, 1, 2, 3], 0.1, 1.0),
        ("uniform", [0.2] * 4, [0, 1, 2, 3], 1e-6, 1.0),
        ("unequal", [0.2, 0.2, 0.2, -0.2], [0, 1, 2, 3], 1e-6, 1.0),
    ]:
        # Indexed and fallback ranks must agree on the collective dtype even with a float64 default.
        torch.set_default_dtype(torch.float64 if name == "mixed_index" else torch.float32)
        for accumulation in [1, 2]:
            accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
            path = root / f"{name}-{accumulation}-{rank}.parquet"
            pd.DataFrame(
                {
                    "index": list(range(8)),
                    "episode_index": [i // 2 for i in range(8)],
                    "progress_sparse": [p for delta in deltas for p in [0.4, 0.4 + delta]],
                }
            ).to_parquet(path)
            config = SampleWeightingConfig(
                type="uniform" if name == "uniform" else "rabc",
                progress_path=str(path),
                epsilon=epsilon,
                extra_params={"fallback_weight": fallback},
            )
            policy, reference = _ScalarPolicy(), _ScalarPolicy()
            optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
            policy, optimizer = accelerator.prepare(policy, optimizer)
            weighter = make_sample_weighter(
                config,
                accelerator.unwrap_model(policy),
                torch.device("cpu"),
                normalization_stats_reducer=partial(accelerator.reduce, reduction="sum"),
            )
            reference_weighter = make_sample_weighter(config, reference, torch.device("cpu"))
            for microbatch in range(accumulation):
                indices = torch.tensor(order) * 2
                targets = torch.tensor([1.0, 1.0, 3.0, 7.0])[order] + microbatch
                full_batch = {"index": indices, "target": targets}
                if name == "missing_index":
                    full_batch = {"target": targets, "action": targets}
                elif name == "mixed_index":
                    # Rank 0 falls back to ones. Absent reference indices use the same raw fallback.
                    full_batch["index"] = torch.tensor([-1, -2, 4, 6])
                weights, _ = reference_weighter.compute_batch_weights(full_batch)
                reference_loss = ((reference.weight - targets).square() * weights).sum() / (
                    weights.sum() + 1e-6
                )
                (reference_loss / accumulation).backward()

                selection = (
                    slice(0, 1)
                    if name == "unequal" and rank == 0
                    else (slice(1, 4) if name == "unequal" else slice(rank * 2, rank * 2 + 2))
                )
                batch = {key: value[selection] for key, value in full_batch.items()}
                if name == "mixed_index" and rank == 0:
                    batch = {"target": batch["target"], "observation.state": batch["target"]}
                tracker = MetricsTracker(
                    len(batch["target"]),
                    4,
                    4,
                    {key: AverageMeter(key, ":.6f") for key in ["loss", "lr", "update_s"]},
                )
                update_policy(tracker, policy, batch, optimizer, 0.0, accelerator, sample_weighter=weighter)
            reference_optimizer.step()
            actual = accelerator.unwrap_model(policy).weight.detach()
            print(
                name,
                accumulation,
                rank,
                "actual",
                actual.item(),
                "reference",
                reference.weight.item(),
                flush=True,
            )
            torch.testing.assert_close(actual, reference.weight, rtol=2e-6, atol=1e-7)
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo required")
def test_weighted_updates_match_the_combined_microbatch(tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.spawn(_weighted_worker, args=(port, str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    while not context.join(timeout=1):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            for process in context.processes:
                process.join(timeout=5)
            raise TimeoutError("Weighted CPU/Gloo update did not complete within 120 seconds")
