# Adding a New Benchmark

This guide walks you through adding a new simulation benchmark to LeRobot. Follow the steps in order and use the existing benchmarks as templates.

A benchmark in LeRobot is a set of [Gymnasium](https://gymnasium.farama.org/) environments that wrap a third-party simulator (like LIBERO or Meta-World) behind a standard `gym.Env` interface. The `lerobot-eval` CLI then runs evaluation uniformly across all benchmarks.

## Existing benchmarks at a glance

Before diving in, here is what is already integrated:

| Benchmark      | Env file            | Config class       | Tasks               | Action dim   | Processor                    |
| -------------- | ------------------- | ------------------ | ------------------- | ------------ | ---------------------------- |
| LIBERO         | `envs/libero.py`    | `LiberoEnv`        | 130 across 5 suites | 7            | `LiberoProcessorStep`        |
| Meta-World     | `envs/metaworld.py` | `MetaworldEnv`     | 50 (MT50)           | 4            | None                         |
| IsaacLab Arena | Hub-hosted          | `IsaaclabArenaEnv` | Configurable        | Configurable | `IsaaclabArenaProcessorStep` |

Use `src/lerobot/envs/libero.py` and `src/lerobot/envs/metaworld.py` as reference implementations.

## How it all fits together

### Data flow

During evaluation, data moves through four stages:

```
1. gym.Env  ──→  raw observations (numpy dicts)

2. Preprocessing  ──→  standard LeRobot keys + task description
   (preprocess_observation, add_envs_task in envs/utils.py)

3. Processors  ──→  env-specific then policy-specific transforms
   (env_preprocessor, policy_preprocessor)

4. Policy  ──→  select_action()  ──→  action tensor
   then reverse: policy_postprocessor → env_postprocessor → numpy action → env.step()
```

Most benchmarks only need to care about stage 1 (producing observations in the right format) and optionally stage 3 (if env-specific transforms are needed).

### Environment structure

`make_env()` returns a nested dict of vectorized environments:

```python
dict[str, dict[int, gym.vector.VectorEnv]]
#    ^suite       ^task_id
```

A single-task env (e.g. PushT) looks like `{"pusht": {0: vec_env}}`.
A multi-task benchmark (e.g. LIBERO) looks like `{"libero_spatial": {0: vec0, 1: vec1, ...}, ...}`.

### How evaluation runs

All benchmarks are evaluated the same way by `lerobot-eval`:

1. `make_env()` builds the nested `{suite: {task_id: VectorEnv}}` dict.
2. `eval_policy_all()` iterates over every suite and task.
3. For each task, it runs `n_episodes` rollouts via `rollout()`.
4. Results are aggregated hierarchically: episode, task, suite, overall.
5. Metrics include `pc_success` (success rate), `avg_sum_reward`, and `avg_max_reward`.

The critical piece: your env must return `info["is_success"]` on every `step()` call. This is how the eval loop knows whether a task was completed.

## What your environment must provide

LeRobot does not enforce a strict observation schema. Instead it relies on a set of conventions that all benchmarks follow.

### Env attributes

Your `gym.Env` must set these attributes:

| Attribute            | Type  | Why                                                  |
| -------------------- | ----- | ---------------------------------------------------- |
| `_max_episode_steps` | `int` | `rollout()` uses this to cap episode length          |
| `task_description`   | `str` | Passed to VLA policies as a language instruction     |
| `task`               | `str` | Fallback identifier if `task_description` is not set |

### Success reporting

Your `step()` and `reset()` must include `"is_success"` in the `info` dict:

```python
info = {"is_success": True}   # or False
return observation, reward, terminated, truncated, info
```

### Observations

The simplest approach is to map your simulator's outputs to the standard keys that `preprocess_observation()` already understands. Do this inside your `gym.Env` (e.g. in a `_format_raw_obs()` helper):

| Your env should output    | LeRobot maps it to         | What it is                            |
| ------------------------- | -------------------------- | ------------------------------------- |
| `"pixels"` (single array) | `observation.image`        | Single camera image, HWC uint8        |
| `"pixels"` (dict)         | `observation.images.<cam>` | Multiple cameras, each HWC uint8      |
| `"agent_pos"`             | `observation.state`        | Proprioceptive state vector           |
| `"environment_state"`     | `observation.env_state`    | Full environment state (e.g. PushT)   |
| `"robot_state"`           | `observation.robot_state`  | Nested robot state dict (e.g. LIBERO) |

If your simulator uses different key names, you have two options:

1. **Recommended:** Rename them to the standard keys inside your `gym.Env` wrapper.
2. **Alternative:** Write an env processor to transform observations after `preprocess_observation()` runs (see step 4 below).

### Actions

Actions are continuous numpy arrays in a `gym.spaces.Box`. The dimensionality depends on your benchmark (7 for LIBERO, 4 for Meta-World, etc.). Policies adapt to different action dimensions through their `input_features` / `output_features` config.

### Feature declaration

Each `EnvConfig` subclass declares two dicts that tell the policy what to expect:

- `features` — maps feature names to `PolicyFeature(type, shape)` (e.g. action dim, image shape).
- `features_map` — maps raw observation keys to LeRobot convention keys (e.g. `"agent_pos"` to `"observation.state"`).

## Step by step

<Tip>
  At minimum, you need three files: a **gym.Env wrapper**, an **EnvConfig
  subclass**, and a **factory dispatch branch**. Everything else is optional or
  documentation.
</Tip>

### Checklist

| File                                     | Required | Why                                       |
| ---------------------------------------- | -------- | ----------------------------------------- |
| `src/lerobot/envs/<benchmark>.py`        | Yes      | Wraps the simulator as a standard gym.Env |
| `src/lerobot/envs/configs.py`            | Yes      | Registers your benchmark for the CLI      |
| `src/lerobot/envs/factory.py`            | Yes      | Tells `make_env()` how to build your envs |
| `src/lerobot/processor/env_processor.py` | Optional | Custom observation/action transforms      |
| `src/lerobot/envs/utils.py`              | Optional | Only if you need new raw observation keys |
| `pyproject.toml`                         | Yes      | Declares benchmark-specific dependencies  |
| `docs/source/<benchmark>.mdx`            | Yes      | User-facing documentation page            |
| `docs/source/_toctree.yml`               | Yes      | Adds your page to the docs sidebar        |

### 1. The gym.Env wrapper (`src/lerobot/envs/<benchmark>.py`)

Create a `gym.Env` subclass that wraps the third-party simulator:

```python
class MyBenchmarkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": <fps>}

    def __init__(self, task_suite, task_id, ...):
        super().__init__()
        self.task = <task_name_string>
        self.task_description = <natural_language_instruction>
        self._max_episode_steps = <max_steps>
        self.observation_space = spaces.Dict({...})
        self.action_space = spaces.Box(low=..., high=..., shape=(...,), dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        ...  # return (observation, info) — info must contain {"is_success": False}

    def step(self, action: np.ndarray):
        ...  # return (obs, reward, terminated, truncated, info) — info must contain {"is_success": <bool>}

    def render(self):
        ...  # return RGB image as numpy array

    def close(self):
        ...
```

Also provide a factory function that returns the nested dict structure:

```python
def create_mybenchmark_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict | None = None,
    env_cls: type | None = None,
) -> dict[str, dict[int, Any]]:
    """Create {suite_name: {task_id: VectorEnv}} for MyBenchmark."""
    ...
```

See `create_libero_envs()` (multi-suite, multi-task) and `create_metaworld_envs()` (difficulty-grouped tasks) for reference.

### 2. The config (`src/lerobot/envs/configs.py`)

Register a config dataclass so users can select your benchmark with `--env.type=<name>`:

```python
@EnvConfig.register_subclass("<benchmark_name>")
@dataclass
class MyBenchmarkEnvConfig(EnvConfig):
    task: str = "<default_task>"
    fps: int = <fps>
    obs_type: str = "pixels_agent_pos"

    features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(<action_dim>,)),
    })
    features_map: dict[str, str] = field(default_factory=lambda: {
        ACTION: ACTION,
        "agent_pos": OBS_STATE,
        "pixels": OBS_IMAGE,
    })

    def __post_init__(self):
        ...  # populate features based on obs_type

    @property
    def gym_kwargs(self) -> dict:
        return {"obs_type": self.obs_type, "render_mode": self.render_mode}
```

Key points:

- The `register_subclass` name is what users pass on the CLI (`--env.type=<name>`).
- `features` tells the policy what the environment produces.
- `features_map` maps raw observation keys to LeRobot convention keys.

### 3. The factory dispatch (`src/lerobot/envs/factory.py`)

Add a branch in `make_env()` to call your factory function:

```python
elif "<benchmark_name>" in cfg.type:
    from lerobot.envs.<benchmark> import create_<benchmark>_envs

    if cfg.task is None:
        raise ValueError("<BenchmarkName> requires a task to be specified")

    return create_<benchmark>_envs(
        task=cfg.task,
        n_envs=n_envs,
        gym_kwargs=cfg.gym_kwargs,
        env_cls=env_cls,
    )
```

If your benchmark needs an env processor, add it in `make_env_pre_post_processors()`:

```python
if isinstance(env_cfg, MyBenchmarkEnvConfig) or "<benchmark_name>" in env_cfg.type:
    preprocessor_steps.append(MyBenchmarkProcessorStep())
```

### 4. Env processor (optional — `src/lerobot/processor/env_processor.py`)

Only needed if your benchmark requires observation transforms beyond what `preprocess_observation()` handles (e.g. image flipping, coordinate conversion):

```python
@dataclass
@ProcessorStepRegistry.register(name="<benchmark>_processor")
class MyBenchmarkProcessorStep(ObservationProcessorStep):
    def _process_observation(self, observation):
        processed = observation.copy()
        # your transforms here
        return processed

    def transform_features(self, features):
        return features  # update if shapes change

    def observation(self, observation):
        return self._process_observation(observation)
```

See `LiberoProcessorStep` for a full example (image rotation, quaternion-to-axis-angle conversion).

### 5. Dependencies (`pyproject.toml`)

Add a new optional-dependency group:

```toml
mybenchmark = ["my-benchmark-pkg==1.2.3", "lerobot[scipy-dep]"]
```

Pinning rules:

- **Always pin** benchmark packages to exact versions for reproducibility (e.g. `metaworld==3.0.0`).
- **Add platform markers** when needed (e.g. `; sys_platform == 'linux'`).
- **Pin fragile transitive deps** if known (e.g. `gymnasium==1.1.0` for Meta-World).
- **Document constraints** in your benchmark doc page.

Users install with:

```bash
pip install -e ".[mybenchmark]"
```

### 6. Documentation (`docs/source/<benchmark>.mdx`)

Write a user-facing page following the template in the next section. See `docs/source/libero.mdx` and `docs/source/metaworld.mdx` for full examples.

### 7. Table of contents (`docs/source/_toctree.yml`)

Add your benchmark to the "Benchmarks" section:

```yaml
- sections:
    - local: libero
      title: LIBERO
    - local: metaworld
      title: Meta-World
    - local: envhub_isaaclab_arena
      title: NVIDIA IsaacLab Arena Environments
    - local: <your_benchmark>
      title: <Your Benchmark Name>
  title: "Benchmarks"
```

## Verifying your integration

After completing the steps above, confirm that everything works:

1. **Install** — `pip install -e ".[mybenchmark]"` and verify the dependency group installs cleanly.
2. **Smoke test env creation** — call `make_env()` with your config in Python, check that the returned dict has the expected `{suite: {task_id: VectorEnv}}` shape, and that `reset()` returns observations with the right keys.
3. **Run a full eval** — `lerobot-eval --env.type=<name> --env.task=<task> --eval.n_episodes=1 --eval.batch_size=1 --policy.path=<any_compatible_policy>` to exercise the full pipeline end-to-end.
4. **Check success detection** — verify that `info["is_success"]` flips to `True` when the task is actually completed. This is what the eval loop uses to compute success rates.

## Writing a benchmark doc page

Each benchmark `.mdx` page should include:

- **Title and description** — 1-2 paragraphs on what the benchmark tests and why it matters.
- **Links** — paper, GitHub repo, project website (if available).
- **Overview image or GIF.**
- **Available tasks** — table of task suites with counts and brief descriptions.
- **Installation** — `pip install -e ".[<benchmark>]"` plus any extra steps (env vars, system packages).
- **Evaluation** — recommended `lerobot-eval` command with `n_episodes` and `batch_size` for reproducible results. Include single-task and multi-task examples if applicable.
- **Policy inputs and outputs** — observation keys with shapes, action space description.
- **Recommended evaluation episodes** — how many episodes per task is standard.
- **Training** — example `lerobot-train` command.
- **Reproducing published results** — link to pretrained model, eval command, results table (if available).

See `docs/source/libero.mdx` and `docs/source/metaworld.mdx` for complete examples.
