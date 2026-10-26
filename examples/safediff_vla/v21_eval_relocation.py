"""[StagePlan-v2.1] (17) Object-relocation test, SOURCE-DOMAIN pilot VAL scenes only: same instruction, target (and for poker its card holder) is moved in the
simulator by a random planar offset to a collision-free spot, the scene is re-rendered, and the predicted grasp displacement is compared with the actual
target displacement. Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/v21_eval_relocation.py --head v2|<dir> --tasks select_poker select_mahjong"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from v21_common import OUT, make_batch, load_policy  # noqa: E402
from v21_eval_offline import load_head  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import robot_base  # noqa: E402
from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402

MIN_SEP = 0.08


def free_joint(entity, physics):
    """qpos slice [x,y,z,quat] of the entity body's free joint (entity bodies are attached under a free joint by the task)."""
    bid = physics.bind(entity.mjcf_model.worldbody).element_id; m = physics.model
    if m.body_jntnum[bid] < 1 or m.jnt_type[m.body_jntadr[bid]] != 0: raise RuntimeError("no free joint")
    adr = int(m.jnt_qposadr[m.body_jntadr[bid]])
    class _J:  # tiny view so qpos[:3] += d works on the live data array
        qpos = physics.data.qpos[adr:adr + 7]
    return _J


def images(e):
    px = e._get_obs()["pixels"]; return np.stack([cv2.resize(np.asarray(px[k]).squeeze(), (224, 224), interpolation=cv2.INTER_AREA) for k in ("image", "second_image", "wrist_image")])


@torch.no_grad()
def predict(pol, pre, head, imgs, state, text):
    r = {"imgs224": imgs, "state": state}; bb = pre(make_batch(r, text)); tok, pad, mod = pol._encode_multimodal_latent(bb); o = head(tok, pad, mod, pol._current_state(bb))
    return (o["grasp_xyz"] * pol.action_pos_std + pol.action_pos_mean)[0].cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="v2"); ap.add_argument("--tasks", nargs="+", default=["select_poker", "select_mahjong", "select_fruit", "select_toy"]); ap.add_argument("--k", type=int, default=4); ap.add_argument("--out", default=None); a = ap.parse_args()
    rows = pickle.load(open(OUT / "referent_table.pkl", "rb")); split = json.load(open(OUT / "pilot_scene_split.json")); rng = np.random.default_rng(0)
    pol, pre = load_policy(); head = load_head(a.head, pol._multimodal_latent_dim()); res = {}
    for task in a.tasks:
        env = make_env(VLABenchEnv(task=task), n_envs=1, use_async_envs=False)[task][0]; G, P = [], []
        for r in [r for r in rows if r["task"] == task and split[r["scene_id"]] == "val"]:
            obs, _ = env.reset(seed=[r["seed"]]); e = env.unwrapped.envs[0]; t = e._env.task; ph = e._env.physics; base = robot_base(e); tgt = t.entities[t.target_entity]
            state = np.asarray(obs["agent_pos"])[0]; text = e.task_description; im0 = images(e); p0 = predict(pol, pre, head, im0, state, text); a0 = np.array(tgt.get_grasped_keypoints(ph)[0], float)
            others = [np.asarray(en.get_xpos(ph), float) for k, en in t.entities.items() if en is not tgt and k != "table" and not k.startswith(("card_holder", "placemat"))]
            holders = [(k, en) for k, en in t.entities.items() if k.startswith("card_holder")] if task == "select_poker" else []
            jt = free_joint(tgt, ph); q0 = np.array(jt.qpos, float).copy(); jh = None
            if holders:
                hk = min(holders, key=lambda kv: np.linalg.norm(np.asarray(kv[1].get_xpos(ph))[:2] - a0[:2])); jh = free_joint(hk[1], ph); qh0 = np.array(jh.qpos, float).copy()
            for _ in range(a.k):
                for _try in range(50):
                    d = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.08, 0.08), 0.0])
                    if np.linalg.norm(d[:2]) < 0.05: continue
                    new = a0 + d
                    if all(np.linalg.norm(new[:2] - o[:2]) > MIN_SEP for o in others) and -0.4 < new[0] - base[0] < 0.4 and 0.25 < new[1] - base[1] < 0.9: break
                else: continue
                jt.qpos[:3] = q0[:3] + d
                if jh is not None: jh.qpos[:3] = qh0[:3] + d
                ph.forward(); imgs = images(e); g = np.asarray(tgt.get_grasped_keypoints(ph)[0], float) - a0; p = predict(pol, pre, head, imgs, state, text) - p0
                G.append(g); P.append(p); jt.qpos[:3] = q0[:3]
                if jh is not None: jh.qpos[:3] = qh0[:3]
                ph.forward()
        env.close(); G, P = np.array(G), np.array(P); ng = np.linalg.norm(G, axis=1)
        cos = (G * P).sum(1) / (ng * np.linalg.norm(P, axis=1) + 1e-9); real = (P * G).sum(1) / ng ** 2
        res[task] = {"n": len(G), "gt_disp_cm": float(ng.mean() * 100), "pred_disp_cm": float(np.linalg.norm(P, axis=1).mean() * 100), "cos": float(cos.mean()), "slope": float((P * G).sum() / (G * G).sum()), "pearson": float(np.corrcoef(G.ravel(), P.ravel())[0, 1]), "realized": float(real.mean()), "cos>0.5": float((cos > 0.5).mean()), "real>0.5": float((real > 0.5).mean())}
        print(task, {k: round(v, 3) for k, v in res[task].items()}, flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__": main()
