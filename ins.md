### Commands (seed=1000, no --wait)

- SmolVLA from scratch on PushT
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com \
  --model smolvla --dataset lerobot/pusht_image --env-type pusht --env-task PushT-v0 \
  --batch-size 64 --steps 200000 --wandb-entity bdaii --wandb-project lerobot-vpl-benchmarks \
  --cpus 32 --gpus 1 --jobs 1 --seed 1000 --lerobot-req 'lerobot==0.3.3'
```

- SmolVLA from scratch on ALOHA cube
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com \
  --model smolvla --dataset lerobot/aloha_sim_transfer_cube_human_image \
  --env-type aloha --env-task AlohaTransferCube-v0 \
  --batch-size 64 --steps 200000 --wandb-entity bdaii --wandb-project lerobot-vpl-benchmarks \
  --cpus 32 --gpus 1 --jobs 1 --seed 1000 --lerobot-req 'lerobot==0.3.3'
```

- Diffusion from scratch on ALOHA cube
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com \
  --model diffusion --dataset lerobot/aloha_sim_transfer_cube_human_image \
  --env-type aloha --env-task AlohaTransferCube-v0 \
  --batch-size 64 --steps 200000 --wandb-entity bdaii --wandb-project lerobot-vpl-benchmarks \
  --cpus 32 --gpus 1 --jobs 1 --seed 1000 --lerobot-req 'lerobot==0.3.3'
```

- PI0 pretrained on ALOHA cube
```bash
python ray_submit_remote.py --ray-url http://tarikmimic.lam-248.ray.clusters.corp.theaiinstitute.com \
  --model pi0 --dataset lerobot/aloha_sim_transfer_cube_human_image \
  --env-type aloha --env-task AlohaTransferCube-v0 \
  --batch-size 16 --steps 200000 --wandb-entity bdaii --wandb-project lerobot-vpl-benchmarks \
  --policy-path 'lerobot/pi0' \
  --cpus 32 --gpus 1 --jobs 1 --seed 1000 --lerobot-req 'lerobot==0.3.3'
```
this
gives
```
WARNING 2025-08-08 00:37:39 ies/utils.py:85 Missing key(s) when loading model: ['normalize_inputs.buffer_observation_state.mean', 'normalize_inputs.buffer_observation_state.std', 'normalize_targets.buffer_action.mean', 'normalize_targets.buffer_action.std', 'unnormalize_outputs.buffer_action.mean', 'unnormalize_outputs.buffer_action.std'] is this problem?
```

- Ensured PI0 uses `transformers==4.52.0` via the job env to avoid Gemma `embed_tokens` error; others unchanged.