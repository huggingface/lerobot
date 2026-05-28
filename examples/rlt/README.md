# PI0.5 + RLT LIBERO Examples

This directory contains utilities for training and evaluating RLT on top of a
frozen PI0.5 policy in LIBERO.

## Workflow

1. Collect or prepare a LeRobotDataset with PI0.5 rollouts.
2. Precompute PI0.5 prefix embeddings and reference actions.
3. Train the RLT token encoder and decoder from the precomputed cache.
4. Run online RLT actor-critic training in LIBERO.
5. Evaluate PI0.5 alone or PI0.5 with an RLT checkpoint.

## Scripts

- `collect_pi05_sim_dataset.py`: collect LIBERO simulation rollouts from a PI0.5 policy into a LeRobotDataset.
- `precompute_pi05_rlt_cache.py`: precompute PI0.5 prefix embeddings for RLT training.
- `train_rlt_token_from_cache.py`: train the RLT token encoder and decoder from cached PI0.5 embeddings.
- `train_rlt_online_libero.py`: run single-process online RLT actor-critic training in LIBERO.
- `evaluate_pi05_rlt_libero_success.py`: evaluate success rate for PI0.5 alone or PI0.5 with RLT.

Run each script with `--help` for available arguments.
