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
    lr = 1e-4
    num_warmup_steps = 500
    num_training_steps = 2000
    record_intervals = 250
    recorded_lrs_at_intervals = [
        2.0e-7,
        5.0200000e-5,
        9.9999890e-5,
        9.3248815e-5,
        7.4909255e-5,
        4.9895280e-5,
        2.4909365e-5,
        6.6464649e-6,
    ]
    optimizer = torch.optim.AdamW(torch.nn.Linear(10, 10).parameters(), lr=lr)

    lr_scheduler = get_scheduler(
        "cosine", optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    assert lr_scheduler.get_last_lr()[0] == 0.0

    for step_idx in range(num_training_steps - record_intervals):
        lr_scheduler.step()
        if step_idx % record_intervals == 0:
            recorded = recorded_lrs_at_intervals.pop(0)
            assert math.isclose(
                lr_scheduler.get_last_lr()[0], recorded, abs_tol=1e-7
            ), f"LR value mismatch at step {step_idx}: {lr_scheduler.get_last_lr()[0]} vs. {recorded}"

    lr_scheduler.step()
    assert math.isclose(
        lr_scheduler.get_last_lr()[0], recorded_lrs_at_intervals[-1], abs_tol=1e-7
    ), f"LR value mismatch at step {num_training_steps}: {lr_scheduler.get_last_lr()[0]} vs. {recorded_lrs_at_intervals[-1]}"


def test_inverse_sqrt_lr_scheduler():
    lr = 1e-4
    num_warmup_steps = 500
    num_training_steps = 2000
    record_intervals = 250
    recorded_lrs_at_intervals = [
        2.0e-7,
        5.02e-5,
        9.9900150e-05,
        8.1595279e-05,
        7.0675349e-05,
        6.3220270e-05,
        5.7715792e-5,
        5.3436983e-5,
    ]
    optimizer = torch.optim.AdamW(torch.nn.Linear(10, 10).parameters(), lr=lr)

    lr_scheduler = get_scheduler(
        "inverse_sqrt", optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    for step_idx in range(num_training_steps - record_intervals):
        lr_scheduler.step()
        if step_idx % record_intervals == 0:
            recorded = recorded_lrs_at_intervals.pop(0)
            assert math.isclose(
                lr_scheduler.get_last_lr()[0], recorded, abs_tol=1e-7
            ), f"LR value mismatch at step {step_idx}: {lr_scheduler.get_last_lr()[0]} vs. {recorded}"

    lr_scheduler.step()
    assert math.isclose(
        lr_scheduler.get_last_lr()[0], recorded_lrs_at_intervals[-1], abs_tol=1e-7
    ), f"LR value mismatch at step {num_training_steps}: {lr_scheduler.get_last_lr()[0]} vs. {recorded_lrs_at_intervals[-1]}"
