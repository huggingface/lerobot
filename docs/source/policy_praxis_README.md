## Praxis-1

Praxis-1 is Runway's open-weight World Action Model for robots: a video pre-trained,
chunk-causal video diffusion backbone (a 27B-parameter mixture-of-experts transformer,
about 3B active per token) with an action expert that denoises 24 Cartesian commands at
15 Hz alongside the next 1.6 s of video of the two rig cameras. The released checkpoint
`praxis-1-droid` (224 x 416 camera tiles with an antialiased
resize; 2 Euler steps, flow shift 12.5, video guidance 4.0) drives a single-arm DROID rig (Franka
arm, exterior and wrist camera). `praxis-1-yam` drives two YAM arms with a top camera and two wrist
cameras. `praxis-1-base` is the start of a fine-tune for a new single-arm rig.

## Project

Announcement: https://runway.com/news/praxis-1-is-now-available

Weights: https://huggingface.co/collections/runwayml/praxis-1 (`praxis-1-droid-lerobot` and the other converted checkpoints)

Code: https://github.com/runwayml/praxis-1-oss

## Simulation benchmark

The `droid_guard` suite of the praxis repository (7 DROID tasks x 8 scenes, MuJoCo), run
through `lerobot-eval` with the `flash` attention
backend: 54 of 56 episodes with `praxis-1-droid`, on LeRobot 0.6's stack and on the LeRobot 0.7 stack
(torch 2.15, transformers 5.22); the praxis repository's own runner also scores 54 of 56 on the
same episodes. Given the same inputs the policy's commands are bitwise equal to the praxis
repository's. See `docs/source/praxis.mdx` for the per-task numbers of both checkpoints and the
paired comparison with the praxis repository's runner.

## Fine-tuning

`lerobot-train` fine-tunes the checkpoint with the praxis objective, in full (FSDP, 4 or more
80 GB GPUs) or with LoRA adapters through PEFT (4 GPUs); `lerobot-eval` loads an adapter on top of
the base, and a merge tool folds it into a plain checkpoint. See the Training section of
`docs/source/praxis.mdx`.
