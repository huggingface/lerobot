import math

import pytest
import torch

from lerobot.common.policies.lr_schedulers import get_scheduler


def test_get_lr_scheduler():
    optimizer = torch.optim.AdamW(torch.nn.Linear(10, 10).parameters(), lr=1e-4)

    lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=500, num_training_steps=2000)
    assert lr_scheduler is not None
    assert lr_scheduler.__class__.__name__ == "LambdaLR"

    lr_scheduler = get_scheduler("inverse_sqrt", optimizer, num_warmup_steps=500, num_training_steps=2000)
    assert lr_scheduler is not None
    assert lr_scheduler.__class__.__name__ == "LambdaLR"

    with pytest.raises(ValueError):
        get_scheduler("invalid", 100, 1000)


def test_cosine_lr_scheduler():
    intervals = 250
    num_warmup_steps = 500
    num_training_steps = 2000
    recorded_lrs_at_intervals = [2.0e-7, 5.0e-5, 1.0e-4, 9.3e-5, 7.5e-5, 5.0e-5, 2.5e-5, 6.6e-6]
    optimizer = torch.optim.AdamW(
        torch.nn.Linear(10, 10).parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6
    )

    lr_scheduler = get_scheduler(
        "cosine", optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    assert lr_scheduler.get_last_lr()[0] == 0.0

    for i in range(num_training_steps):
        optimizer.step()
        lr_scheduler.step()
        if i == 0 or (i + 1) % intervals == 0:
            recorded = recorded_lrs_at_intervals.pop(0)
            assert math.isclose(
                lr_scheduler.get_last_lr()[0], recorded
            ), f"LR value mismatch at step {i}: {lr_scheduler.get_last_lr()[0]} vs. {recorded}"

    assert lr_scheduler.get_last_lr()[0] == recorded_lrs_at_intervals.pop(0)


def test_inverse_sqrt_lr_scheduler():
    intervals = 250
    num_warmup_steps = 500
    num_training_steps = 2000
    recorded_lrs_at_intervals = [2.0e-7, 5.0e-5, 1.0e-4, 8.2e-5, 7.1e-5, 6.3e-5, 5.8e-5, 5.3e-5]
    optimizer = torch.optim.AdamW(
        torch.nn.Linear(10, 10).parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6
    )

    lr_scheduler = get_scheduler(
        "inverse_sqrt", optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    for i in range(num_training_steps):
        lr_scheduler.step()
        if i == 0 or (i + 1) % intervals == 0:
            recorded = recorded_lrs_at_intervals.pop(0)
            assert math.isclose(
                lr_scheduler.get_last_lr()[0], recorded
            ), f"LR value mismatch at step {i}: {lr_scheduler.get_last_lr()[0]} vs. {recorded}"

    assert lr_scheduler.get_last_lr()[0] == recorded_lrs_at_intervals.pop(0)
