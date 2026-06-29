# examples/vllm_policy

A ready-to-use `--policy.path` directory for the `vllm` policy (remote vLLM OpenPI
server), plus helper scripts for regenerating the config and running a single-task
visualized rollout.

The policy is an in-tree lerobot policy at `src/lerobot/policies/vllm/` (registered as
`--policy.type=vllm`, like `act`, `groot`, `pi0`, …).

## Quick start

```bash
# Install lerobot with the vllm extra (pulls in the websockets/msgpack deps):
uv pip install -e ".[vllm]"

# Verify the registration appears:
lerobot-eval --help | grep -E "policy.type" | head -1
# → --policy.type   {... vllm ...}
```

## 1. Run a full eval (LIBERO, real GR00T server on :8000)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/lerobot-eval \
  --policy.path=examples/vllm_policy \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --policy.n_action_steps=10 \
  --output_dir=./outputs/libero_eval
```

## 2. Visualize a single task live (Rerun web viewer)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python \
  examples/vllm_policy/scripts/eval_visualize.py \
  --benchmark libero_object --task-id 0 --open-browser
```

Variants: `--spawn` (native desktop viewer) · `--save /tmp/run.rrd --no-keep-alive`
(offline recording, view with `rerun /tmp/run.rrd`).

## 3. Local mock server (no GPU smoke test)

```bash
.venv/bin/python -m lerobot.policies.vllm.mock_server --host 127.0.0.1 --port 8001
# then re-point the config to the mock:
.venv/bin/python examples/vllm_policy/scripts/generate_policy_dir.py --port 8001
```

## 4. (Re)generate the policy-path JSONs

The three JSONs in this directory (`config.json`, `policy_preprocessor.json`,
`policy_postprocessor.json`) are **generated artifacts**, not hand-written. They are
produced from the `VllmConfig` dataclass defaults by `generate_policy_dir.py`:

```bash
.venv/bin/python examples/vllm_policy/scripts/generate_policy_dir.py
```

Internally the script does:

```python
cfg = VllmConfig(host=..., port=..., chunk_size=..., n_action_steps=...)
cfg.save_pretrained(out_dir)                       # -> config.json
pre, post = make_vllm_pre_post_processors(cfg)
pre.save_pretrained(out_dir, ...)                  # -> policy_preprocessor.json
post.save_pretrained(out_dir, ...)                 # -> policy_postprocessor.json
```

Re-run it whenever you:
- change a field / default in `src/lerobot/policies/vllm/configuration_vllm.py`
  (this keeps the committed JSONs in sync — otherwise they silently go stale), or
- want a directory pointed at a different server / chunking, e.g.:

```bash
.venv/bin/python examples/vllm_policy/scripts/generate_policy_dir.py \
  --out /tmp/my_vllm_policy --host 10.0.0.5 --port 8001 --chunk-size 32 --n-action-steps 16
```

Options: `--out` (default `examples/vllm_policy`), `--host`, `--port`, `--chunk-size`,
`--n-action-steps`.

> You can also skip this directory entirely and run with `--policy.type=vllm` plus CLI
> overrides (e.g. `--policy.host`, `--policy.port`); the `--policy.path` dir is just a
> ready-to-use, version-controlled example.

## Contents

```
examples/vllm_policy/
├── README.md                       (this file)
├── config.json                     # generated: {"type": "vllm", host, port, ...}
├── policy_preprocessor.json        # generated: rename + device steps (no normalization)
├── policy_postprocessor.json       # generated: move action to CPU
└── scripts/
    ├── generate_policy_dir.py      # regenerates the three JSONs above
    └── eval_visualize.py
```

## Files NOT here

| What | Where |
|---|---|
| `vllm` policy source | `src/lerobot/policies/vllm/` |
| Registration | `src/lerobot/policies/__init__.py` + `factory.py` (`vllm` branches) |

## Action / state convention (LIBERO)

Authoritative source: `Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py` +
`examples/LIBERO/modality.json`.

- Embodiment: `libero_sim` (= `LIBERO_PANDA`). Requires the post-trained checkpoint
  `nvidia/GR00T-N1.7-LIBERO/<suite>` on the server (e.g. `.../libero_object`).
- Action: native LIBERO 7-D `[x,y,z,roll,pitch,yaw,gripper]` — fed as **delta** to
  `env.step` (relative control); gripper is `normalize [0,1]→[-1,1]` → `sign` → `invert`.
- Images sent as an ordered list `[agentview, eye_in_hand]` (server feeds HF image
  processor; dict rejected).
- Serialization: vendored OpenPI msgpack-numpy format (`__ndarray__` envelope). The
  standalone `msgpack-numpy` PyPI package is NOT wire-compatible.
- Legacy DROID `eef_9d` mode (SE(3)/rot6d math path) is available via
  `--policy.action_space=eef_9d`.

### Other embodiments

The default decode is a generic **flat concat**: the server returns one trajectory per
key, and the policy concatenates the keys named in `action_keys` (in order) into the env
action vector, slicing the request state via `modality_config["state"]`. A new embodiment
whose action is a flat vector of scalar channels is therefore **config-only** — set
`--policy.embodiment`, `--policy.action_keys`, `--policy.gripper_keys`, and the gripper
flags; no code change. `action_dim` is derived from `len(action_keys)`. Embodiments needing
different action math (like `eef_9d`) get their own decode path in `modeling_vllm.py`.

Tunables: `--policy.host/port`, `--policy.embodiment`, `--policy.action_space`,
`--policy.action_keys`, `--policy.gripper_keys`, `--policy.decode_subtract_state`,
`--policy.gripper_normalize`, `--policy.gripper_binarize`, `--policy.gripper_invert`,
`--policy.action_scale`, `--policy.request_seed`.
