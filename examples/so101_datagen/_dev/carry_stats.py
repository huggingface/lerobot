"""Carry heights over many physics-only episodes: lift, carry peak, clearance over the rim, success.

PYTHONPATH=. python examples/so101_datagen/_dev/carry_stats.py --seeds 400 --out review/0_carry/after.npz
"""

import argparse
import collections

import numpy as np

from examples.so101_datagen.assets import load_meshes
from examples.so101_datagen.config import GenerationConfig
from examples.so101_datagen.episode import EpisodeError, prepare, simulate
from examples.so101_datagen.tasks import TASKS, find, obstacle_top

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=400)
parser.add_argument("--out", default=None)
args = parser.parse_args()

cfg = GenerationConfig()
cfg.tasks.weights = {"place_in": 1.0, "place_near": 1.0}
meshes = load_meshes()
fails = collections.Counter()
ok = collections.Counter()
rows = collections.defaultdict(list)
curves = []
for seed in range(args.seeds):
    try:
        ep = prepare(seed, cfg, meshes)
        simulate(ep, TASKS[ep.task], cfg.fps)
    except EpisodeError as e:
        fails[e.reason] += 1
        continue
    ok[ep.task] += 1
    m = ep.plan.marks
    name = ep.instance.roles["object"]
    obj = find(ep.scene, name)
    target = find(ep.scene, ep.instance.roles["target"])
    body = ep.info.object_bodies[name]
    adr = ep.model.jnt_qposadr[ep.model.body_jntadr[body]]
    bottom = ep.qpos[:, adr + 2] - obj.half_size[-1]
    a, b = m["lifted"], m["release"]
    exclude = {name} if ep.task == "place_in" else {name, target.name}
    top = obstacle_top(ep.scene, obj.pos, ep.qpos[b, adr : adr + 2], exclude)
    carry = bottom[a:b]
    # Clearance where it matters: the lowest the object's bottom gets over the container's walls.
    crossing = np.nan
    if ep.task == "place_in":
        rel = ep.qpos[a:b, adr : adr + 2] - np.asarray(target.pos)
        c, s_ = np.cos(-target.yaw), np.sin(-target.yaw)
        local = rel @ np.array([[c, s_], [-s_, c]])
        outer = target.inner_half + target.thickness + obj.half_size[0]
        inner = target.inner_half - obj.half_size[0]
        over_wall = np.all(np.abs(local) < outer, axis=1) & ~np.all(np.abs(local) < max(inner, 0.0), axis=1)
        if over_wall.any():
            crossing = carry[over_wall].min() - target.wall_height
    rows[ep.task].append(
        (bottom[a], carry.max(), carry.max() - top, crossing, np.argmax(carry) / max(len(carry), 1))
    )
    curves.append((ep.task, carry, top))
print("ok", dict(ok), "failures", dict(fails))


def q(r: np.ndarray, c: int) -> np.ndarray:
    return (np.nanquantile(r[:, c], [0.1, 0.25, 0.5, 0.75, 0.9]) * (100 if c < 4 else 1)).round(1)


for task, r in rows.items():
    r = np.array(r)
    print(task, len(r))
    print("  object bottom at lift (cm)        ", q(r, 0))
    print("  carry peak (cm)                   ", q(r, 1))
    print("  peak over obstacles on path (cm)  ", q(r, 2))
    if task == "place_in":
        print("  lowest over the box walls (cm)    ", q(r, 3), f"(min {np.nanmin(r[:, 3]) * 100:.1f})")
    print("  peak position in carry (0..1)     ", q(r, 4))
if args.out:
    np.save(args.out, np.array(curves, dtype=object), allow_pickle=True)
