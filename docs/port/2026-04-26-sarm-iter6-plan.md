# SARM iter-6 plan (2026-04-26)

beads lerobot-48. extend epic 33.

## goal
fold _3 (new fail modes: 0/4 ×10, 3/4 plateau ×5) into train. iter-5 ckpt's lin_mad/plateau borderline; _3 should harden 0/4 FP + plateau gates. champ if all gates pass on **full val + new-val subset**.

## inputs
| ds | eps | fr | bucket 0/1/2/3/4 |
|---|---|---|---|
| filtered | 51 | 11078 | (in merged_v1) |
| _2 | 52 | 14370 | (in merged_v1) |
| merged_v1 | 103 | 25448 | 18/15/15/15/40 |
| _3 (drop ep11=1/4) | 29 | 11048 | 10/0/14/5/0 |

## merged_v2
filtered ∪ _2 ∪ _3\{ep11} = **132 eps, ~36.5k fr**, bucket **28/15/29/20/40**.

## train/val split
stratified per (bucket × source). seed=1000. ~20% val.

| bucket | n | val | val(_3) |
|---|---|---|---|
| 0 | 28 | 6  | 2 |
| 1 | 15 | 3  | 0 |
| 2 | 29 | 6  | 3 |
| 3 | 20 | 4  | 1 |
| 4 | 40 | 8  | 0 |
| total | 132 | **27** | **6** |

→ train ds `local/sim_assemble_sarm_merged_v2` (105 eps), val ds `local/sim_assemble_sarm_merged_v2_val` (27 eps), new-val subset = val eps ∩ src=_3 (6 eps).

## train
copy `sim_assemble_sarm_ext_iter5_2cam_train.json` → `iter6_2cam_train.json`. only diffs:
- `output_dir: outputs/sim_assemble_sarm_ext_iter6_2cam`
- `dataset.repo_id: local/sim_assemble_sarm_merged_v2`
- (optionally bump steps 5000→7500 since +30 eps)

## eval
1. **full val**: `merged_v2_val` (27 eps) — old gates
2. **new val**: filter to src=_3 (6 eps) — esp 0/4 FP rate (target 0%), 3/4 plateau

reuse `eval_sarm_sim_assemble.py --type=sarm_ext --stats=merged_v2`.

## gates (must pass full + new)
- succ_term≥0.9 ≥95% on 4/4 (only full val has 4/4)
- 0/4 max≥0.5 = 0% (KEY for new — _3 has 10× 0/4)
- fail_term≥0.9 = 0% on partials
- lin_mad ≤0.25, mean_mid ≥0.25, mono ≥0.85 (4/4)
- stage_not_exceed ≥0.9
- plateau ±0.10 (KEY for new — _3 has 5× 3/4)

## iter rule
all pass on full ∧ new → champion. else brainstorm:
- knob sweep (rewind_prob, stage_loss_weight, drop_n_last)
- aug (crop, jitter)
- more steps
- redo split

## overlay viz (final)
`overlay_sarm_video_sim_assemble.py` on _3 src eps from val:
- 2× succ-ish (e.g. 2/4 partials reaching bring_box) — eps **TBD from val list**
- 2× partial (3/4 plateau eps from _3 25–29)
- 1× 0/4 timeout (from _3 15–24)

out `outputs/sarm_videos/iter6/`.

## execution
1. drop ep11 from _3 → cache `local/_3_no_ep11`
2. merge filtered ∪ _2 ∪ _3_no_ep11 → `local/sim_assemble_sarm_merged_v2`
3. stratified split → save val ep ids json `iter6_val_eps.json`
4. write new-val ep ids json (val ∩ src=_3)
5. write `iter6_2cam_train.json`
6. train (bg, log to file)
7. eval full+new
8. overlay if pass
