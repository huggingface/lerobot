import gymnasium_robotics
import gymnasium as gym
import numpy as np
from typing import Dict
from lerobot.envs.configs import GymRoboticsEnv

def create_gymnasium_robotics_envs(
    cfg: GymRoboticsEnv,
    n_envs: int = 1,
    use_async_envs: bool = False,
) -> Dict[str, Dict[int, gym.vector.VectorEnv]]:
    """
    Build vectorized GymRoboticsEnv(s) from the GymRoboticsEnv config and return:
        { "<env_type>": { 0: <VectorEnv> } }
    Minimal and consistent with make_env(...) expected return type.
    """
    # pull minimal fields from the config (with safe defaults)
    task = getattr(cfg, "task", "FetchPickAndPlace-v4")
    base_seed = getattr(cfg, "seed", 0)
    image_key = getattr(cfg, "image_key", "agentview_image")
    episode_length = getattr(cfg, "episode_length", None)
    max_state_dim = getattr(cfg, "max_state_dim", None)

    # per-worker factory functions
    def _mk_one(worker_idx: int):
        def _ctor():
            seed = None if base_seed is None else int(base_seed) + worker_idx
            return GymRoboticsEnv(task=task, seed=seed, image_key=image_key, max_state_dim=max_state_dim, episode_length=episode_length)
        return _ctor

    fns = [_mk_one(i) for i in range(n_envs)]
    vec_env = gym.vector.AsyncVectorEnv(fns) if use_async_envs else gym.vector.SyncVectorEnv(fns)

    # key name kept simple/flat; matches your --env.type
    return {"gymnasium-robotics": {0: vec_env}}

class GymRoboticsEnv(gym.Env):
    """Minimal adapter: wraps a Gymnasium-Robotics env and returns a LeRobot-style obs dict."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self, 
        task: str, 
        seed: int | None = 0, 
        image_key: str = "agentview_image", 
        episode_length: int | None = None,
        max_state_dim: int | None = None, 
        **make_kwargs
    ):
        gym.register_envs(gymnasium_robotics)
        make_kwargs = dict(make_kwargs or {})
        make_kwargs["render_mode"] = "rgb_array"
        self.env = gym.make(task, max_episode_steps=episode_length, **make_kwargs)

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._image_key = image_key
        self._max_state_dim = max_state_dim

        # action space: forward from underlying env
        self.action_space = self.env.action_space

        # --- infer observation space once (do a temp reset+render) ---
        tmp_obs, _ = self.env.reset(seed=int(self._rng.integers(0, 2**31 - 1)) if seed is not None else None)
        frame = self.env.render()
        obs = self._to_obs(tmp_obs, frame)

        # build observation_space to match o
        def _box_like(x, low=-np.inf, high=np.inf, dtype=np.float32):
            x = np.asarray(x)
            return gym.spaces.Box(low=low, high=high, shape=x.shape, dtype=dtype)

        img = obs["images"][self._image_key]
        spaces = {
            "images": gym.spaces.Dict({self._image_key: gym.spaces.Box(low=0, high=255, shape=img.shape, dtype=np.uint8)}),
            "state": _box_like(obs["state"]),
            # NEW — aliases for libero-style preprocessors:
            "agent_pos": _box_like(obs["state"]),
            "pixels": gym.spaces.Box(low=0, high=255, shape=img.shape, dtype=np.uint8),
        }
        if "goal" in obs:
            spaces["goal"] = _box_like(obs["goal"])
        if "achieved_goal" in obs:
            spaces["achieved_goal"] = _box_like(obs["achieved_goal"])

        self.observation_space = gym.spaces.Dict(spaces)
        # leave env in a valid state; vector wrapper will call reset() again later

        # passthrough spec (if present on wrapped env)
        self.spec = getattr(self.env, "spec", None)

        max_steps = episode_length
        if max_steps is None:
            # determine max episode steps for upstream code that reads _max_episode_steps
            max_steps = getattr(self.env, "_max_episode_steps", None)
            if max_steps is None and self.spec is not None:
                max_steps = getattr(self.spec, "max_episode_steps", None)

            # try unwrapping one level if wrapped
            if max_steps is None and hasattr(self.env, "env"):
                inner = getattr(self.env, "env")
                max_steps = getattr(inner, "_max_episode_steps", None)
                if max_steps is None:
                    inner_spec = getattr(inner, "spec", None)
                    if inner_spec is not None:
                        max_steps = getattr(inner_spec, "max_episode_steps", None)

            # final fallback
            if max_steps is None:
                max_steps = 1000  # sensible default; adjust if you prefer

        self._max_episode_steps = int(max_steps)


    def reset(self, seed: int | None = None, **kwargs):
        if seed is None and self._seed is not None:
            seed = int(self._rng.integers(0, 2**31 - 1))
        super().reset(seed=seed)
        tmp_obs, info = self.env.reset(seed=seed)
        frame = self.env.render()
        observation = self._to_obs(tmp_obs, frame)
        return observation, info

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Box):
            action = np.clip(np.asarray(action, dtype=np.float32),
                             self.action_space.low, self.action_space.high)
        tmp_obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        obs_out = self._to_obs(tmp_obs, frame)
        return obs_out, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()

    def render(self):
        """Return an RGB frame (HxWx3, uint8) like Gymnasium expects."""
        frame = self.env.render()  # underlying env created with render_mode='rgb_array'
        if frame is None:
            raise RuntimeError("render() returned None; ensure render_mode='rgb_array' in make().")
        return frame.astype(np.uint8, copy=False)

    # ---- helpers ----
    @staticmethod
    def _flat(x):
        if x is None: return np.zeros((0,), dtype=np.float32)
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def _to_obs(self, obs, frame):
        if isinstance(obs, dict):
            state = self._flat(obs.get("observation"))
            desired = obs.get("desired_goal")
            achieved = obs.get("achieved_goal")
            rgb = frame.astype(np.uint8, copy=False)
        elif isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] in (1, 3):
            # Atari-style ndarray: treat as IMAGE, not state
            # use obs as the frame if frame is None
            rgb_src = frame if frame is not None else obs
            rgb = rgb_src.astype(np.uint8, copy=False)
            # no structured state in Atari pixels; provide empty state vector
            state = np.empty((0,), dtype=np.float32)
            desired = achieved = None
        else:
            # fallback: unknown non-dict obs → treat as state only
            state = self._flat(obs)
            if self._max_state_dim is not None and len(state) > self._max_state_dim:
                state = state[:self._max_state_dim]
            desired = achieved = None
            rgb = frame.astype(np.uint8, copy=False)

        rgb = frame.astype(np.uint8, copy=False)

        out = {
            # gym original keys
            "images": {self._image_key: rgb},
            "state": state,
            # aliases expected by LeRobot preprocessors
            "agent_pos": state,   # alias for state
            "pixels": rgb,        # alias for a single RGB view
        }
        if desired is not None:  out["goal"] = self._flat(desired)
        if achieved is not None: out["achieved_goal"] = self._flat(achieved)
        return out
