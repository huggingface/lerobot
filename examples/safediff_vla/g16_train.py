#!/usr/bin/env python
"""G16-E: train ONLY the GeometryConditioner (~0.2M) on the original 6-task data: policy.forward's own action objective + 0.1 * L_progress (PRE_GRASP only). VLM / decoder / target branch / grounded feature / phase / gripper frozen. seed 0, 2000 steps, same held-out split as G11 (seed-0 perm, 10%).
Pre-fixed selection rule: the FINAL step is used; it is flagged (not replaced) if held-out position loss is not below V2's or held-out gripper loss is >5% worse than V2's.
alpha_max: fixed once = 0.5 * mean ||target_xyz_encoder(target_xyz)|| / sqrt(512) (per-element tanh bound => conditioner output norm <= 50% of the mean target-branch norm)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import g16_geometry as G  # noqa: E402
from _multitask_common import RENAME_MAP  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

OUT = Path("outputs/train/safediff_vla_g16"); OUT.mkdir(parents=True, exist_ok=True); TASKS = ["select_poker", "select_mahjong", "select_fruit", "select_drink", "select_toy", "select_book"]; LAMBDA_PROGRESS = 0.1; ALPHA_FRAC = 0.5


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); E = E[E.task.isin(TASKS)]; eps = sorted(E.episode.astype(int).tolist()); perm = rng.permutation(eps); hold = set(perm[:int(0.1 * len(eps))].tolist()); tr_eps = [e for e in eps if e not in hold]; ho_eps = sorted(hold)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda"); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    def loader(ep_list, bs):
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", episodes=ep_list), policy=policy.config, rename_map=RENAME_MAP, batch_size=bs); ds = make_dataset(cfg)
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=6, drop_last=True)
    def prep(b):
        for k in list(b):
            if k.startswith("observation.images") and b[k].dtype == torch.uint8: b[k] = b[k].float() / 255.0
        return pre(b)
    tl = loader(tr_eps, a.bs); hl = loader(ho_eps, a.bs)
    # fixed alpha_max from the existing target branch's mean output norm
    norms = []; it = iter(tl); enc = policy.decoder.target_xyz_encoder
    with torch.no_grad():
        for _ in range(20):
            b = prep(next(it)); lt, lp, lm = policy._encode_multimodal_latent(b); tq = policy.target_query_extractor(lt, lp, lm); f = policy.visual_grounding_cross_attn(tq, lt, lp, lm); norms.append(enc(policy.target_point_head(f)).norm(dim=-1).mean().item())
    alpha = ALPHA_FRAC * float(np.mean(norms)) / np.sqrt(policy.decoder.action_head.in_features); print(f"mean target-branch norm {np.mean(norms):.3f} -> alpha_max {alpha:.5f}", flush=True)
    geo = G.install(policy, alpha); geo.train(); n_par = sum(p.numel() for p in geo.parameters()); print("trainable params:", n_par, flush=True); opt = torch.optim.AdamW(geo.parameters(), lr=a.lr, weight_decay=1e-4)
    HB = []; _it = iter(hl)
    for _ in range(60): HB.append(prep(next(_it)))
    @torch.no_grad()
    def evaluate():
        out = {"on": [], "off": []}; geo.eval()
        for b in HB:
            for mode in ("on", "off"):
                policy.decoder.geo_on = mode == "on"; loss, m = policy.forward(b); pl = G.progress_loss(policy.decoder).item(); out[mode].append((loss.item(), m["loss_pos"], m["loss_rot"], m["loss_grip"], pl))
        policy.decoder.geo_on = True; geo.train(); return {k: np.mean(v, axis=0).round(5).tolist() for k, v in out.items()}
    e0 = evaluate(); print("held-out BEFORE (loss/pos/rot/grip/L_progress):", e0, flush=True); log = []; t0 = time.time(); it = iter(tl); step = 0
    while step < a.steps:
        try: b = prep(next(it))
        except StopIteration: it = iter(tl); continue
        loss, m = policy.forward(b); lp = G.progress_loss(policy.decoder); tot = loss + LAMBDA_PROGRESS * lp; opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(geo.parameters(), 1.0); opt.step(); step += 1
        if step % 25 == 0: log.append({"step": step, "loss": m["loss"], "loss_pos": m["loss_pos"], "loss_grip": m["loss_grip"], "L_progress": lp.item()})
        if step % a.eval_every == 0 or step == a.steps: ev = evaluate(); print(f"step {step} train pos {m['loss_pos']:.5f} Lprog {lp.item():.4f} | held-out ON {ev['on']} | OFF(V2) {ev['off']} | {time.time() - t0:.0f}s", flush=True)
    fin = evaluate(); flag = not (fin["on"][1] < fin["off"][1] and fin["on"][3] <= 1.05 * fin["off"][3]); torch.save({"state": geo.state_dict(), "alpha_max": alpha}, OUT / "geometry_conditioner.pt")
    (OUT / "g16_train_log.json").write_text(json.dumps({"params": n_par, "alpha_max": alpha, "steps": a.steps, "before": e0, "after": fin, "retention_flag": flag, "log": log, "n_train_eps": len(tr_eps)}, indent=1)); print("FINAL held-out", fin, "| retention flag:", flag, flush=True)


if __name__ == "__main__":
    main()
