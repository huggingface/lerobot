"""Review clips: physics-only episodes rendered from the front camera and a side witness camera.

Each seed gives one mp4 (front | side view of the carry, captions burned in) and a height-vs-time strip
of the held object's bottom with the obstacle rim. Run from the datagen worktree:

    PYTHONPATH=. MUJOCO_GL=egl python examples/so101_datagen/_dev/review_clips.py --seeds 0 1 2 --out review/0_carry/after
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import av  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from examples.so101_datagen.assets import load_meshes  # noqa: E402
from examples.so101_datagen.config import GenerationConfig  # noqa: E402
from examples.so101_datagen.episode import EpisodeError, prepare, simulate  # noqa: E402
from examples.so101_datagen.tasks import TASKS, find  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tasks", nargs="+", default=None, help="restrict the task mix")
parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
args = parser.parse_args()

cfg = GenerationConfig()
if args.tasks:
    cfg.tasks.weights = dict.fromkeys(args.tasks, 1.0)
meshes = load_meshes()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
H, W = 360, 480


def caption(image: np.ndarray, lines: list[str]) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for k, text in enumerate(lines):
        draw.rectangle([0, 14 * k, 7 * len(text) + 6, 14 * k + 14], fill=(0, 0, 0))
        draw.text((3, 14 * k + 1), text, fill=(255, 255, 0))
    return np.asarray(pil)


def write(path: Path, frames: list[np.ndarray], fps: float) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=int(round(fps)))
        stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for f in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        container.mux(stream.encode())


for seed in args.seeds:
    try:
        ep = prepare(seed, cfg, meshes)
        simulate(ep, TASKS[ep.task], cfg.fps)
    except EpisodeError as e:
        print(f"seed {seed}: failed ({e.reason})")
        continue
    model = ep.model
    model.vis.global_.offwidth, model.vis.global_.offheight = max(W, 640), max(H, 480)
    renderer = mujoco.Renderer(model, H, W)
    data = mujoco.MjData(model)
    name = ep.instance.roles["object"]
    obj = find(ep.scene, name)
    body = ep.info.object_bodies[name]
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    target = find(ep.scene, ep.instance.roles["target"]) if "target" in ep.instance.roles else None
    rim = target.wall_height if target is not None and hasattr(target, "wall_height") else 0.0
    # Side witness: perpendicular to the line from the object to its destination, at table height.
    a = np.asarray(obj.pos[:2])
    b = np.asarray(target.pos[:2]) if target is not None else a + np.array([0.0, 0.1])
    mid = (a + b) / 2
    along = (b - a) / max(np.linalg.norm(b - a), 1e-6)
    side = mujoco.MjvCamera()
    side.lookat[:] = [mid[0], mid[1], 0.06]
    side.distance = 0.55
    side.elevation = -8
    # Of the two sides of the carry, the one away from the robot base, so the arm doesn't hide it.
    heading = float(np.rad2deg(np.arctan2(along[1], along[0])))
    candidates = []
    for azimuth in (heading + 90, heading - 90):
        az = np.deg2rad(azimuth)
        pos = np.asarray(side.lookat[:2]) - side.distance * np.array([np.cos(az), np.sin(az)])
        candidates.append((np.linalg.norm(pos), azimuth))
    side.azimuth = max(candidates)[1]
    labels = [ep.plan.subtasks[i] for i in ep.plan.subtask_of_frame]
    frames, bottoms = [], []
    for t, qpos in enumerate(ep.qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="front")
        front = renderer.render()
        renderer.update_scene(data, camera=side)
        witness = renderer.render()
        bottom = qpos[adr + 2] - obj.half_size[-1]
        bottoms.append(bottom)
        frames.append(
            np.concatenate(
                [
                    caption(front, [f"seed {seed} {ep.task}", labels[t]]),
                    caption(witness, [f"object bottom {bottom * 100:4.1f} cm", f"rim {rim * 100:.1f} cm"]),
                ],
                1,
            )
        )
    renderer.close()
    write(out / f"seed{seed}_{ep.task}.mp4", frames, cfg.fps * args.speed)
    bottoms = np.asarray(bottoms)
    m = ep.plan.marks
    np.savez(
        out / f"seed{seed}.npz",
        bottom=bottoms,
        rim=rim,
        lifted=m.get("lifted", -1),
        release=m.get("release", -1),
    )
    print(f"seed {seed}: {ep.task}, carry peak {bottoms.max() * 100:.1f} cm, rim {rim * 100:.1f} cm -> {out}")
