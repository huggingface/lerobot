#!/usr/bin/env python
"""G11-C: train ONLY the ResidualConditioner (~0.2M params) on the original 6-task data with the policy's own decoder action objective (policy.forward loss). VLM / decoder / target_point_head / grounded modules frozen. No gripper change, no new auxiliary targets.
Held-out episodes (10%) give the before/after action-loss comparison (position MSE in normalised units, per task)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import g11_residual as R  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT = Path("outputs/train/safediff_vla_g11"); OUT.mkdir(parents=True, exist_ok=True); TASKS = ["select_poker", "select_mahjong", "select_fruit", "select_drink", "select_toy", "select_book"]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); E = E[E.task.isin(TASKS)]; eps = sorted(E.episode.astype(int).tolist()); perm = rng.permutation(eps); hold = set(perm[:int(0.1 * len(eps))].tolist()); tr_eps = [e for e in eps if e not in hold]; ho_eps = sorted(hold)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda"); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    res = R.install(policy); res.train(); n_par = sum(p.numel() for p in res.parameters()); print("trainable params:", n_par, flush=True)
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    def loader(ep_list, shuffle, bs):
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", episodes=ep_list), policy=policy.config, rename_map=RENAME_MAP, batch_size=bs); ds = make_dataset(cfg)
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=6, drop_last=True, persistent_workers=False, pin_memory=False), ds
    def prep(b):
        for k in list(b):
            if k.startswith("observation.images") and b[k].dtype == torch.uint8: b[k] = b[k].float() / 255.0
        return pre(b)
    tl, ds = loader(tr_eps, True, a.bs); hl, _ = loader(ho_eps, True, a.bs); opt = torch.optim.AdamW(res.parameters(), lr=a.lr, weight_decay=1e-4); print(f"train episodes {len(tr_eps)} / held-out {len(ho_eps)}", flush=True)

    HB = []; _it = iter(hl)
    for _ in range(80): HB.append(prep(next(_it)))  # fixed held-out batches (same for every evaluation)

    @torch.no_grad()
    def evaluate(n_batches=40):
        out = {"on": [], "off": []}; res.eval()
        for b in HB[:n_batches]:
            for mode in ("on", "off"):
                policy.decoder.residual_on = mode == "on"; loss, m = policy.forward(b); out[mode].append((loss.item(), m["loss_pos"], m["loss_rot"], m["loss_grip"]))
        policy.decoder.residual_on = True; res.train(); return {k: np.mean(v, axis=0).round(5).tolist() for k, v in out.items()}
    log = []; e0 = evaluate(); print("held-out BEFORE training (untrained conditioner == V2): loss/pos/rot/grip", e0, flush=True); t0 = time.time(); it = iter(tl); step = 0
    while step < a.steps:
        try: b = prep(next(it))
        except StopIteration: it = iter(tl); continue
        loss, m = policy.forward(b); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(res.parameters(), 1.0); opt.step(); step += 1
        if step % 25 == 0: log.append({"step": step, "loss": m["loss"], "loss_pos": m["loss_pos"], "loss_grip": m["loss_grip"]})
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(); print(f"step {step} train loss {m['loss']:.5f} pos {m['loss_pos']:.5f} | held-out residual ON {ev['on']} | OFF(V2) {ev['off']} | {time.time() - t0:.0f}s", flush=True)
    torch.save(res.state_dict(), OUT / "residual_conditioner.pt"); (OUT / "g11_train_log.json").write_text(json.dumps({"params": n_par, "steps": a.steps, "before": e0, "after": evaluate(80), "log": log, "n_train_eps": len(tr_eps)}, indent=1)); print("saved", OUT / "residual_conditioner.pt")


if __name__ == "__main__":
    main()
