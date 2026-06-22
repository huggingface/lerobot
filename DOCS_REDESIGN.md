# LeRobot Documentation Redesign — Proposal

**Status:** Draft for maintainer review · **Author:** Dev Rel · **Date:** June 2026
**Scope:** Information architecture + UX redesign of https://huggingface.co/docs/lerobot (source: `docs/source/`)

---

## 1. Executive summary

**The problem, in five points:**

1. **No guided beginner path.** "Get started" is three pages (landing, install, CLI cheat-sheet). After that, a newcomer lands in a "Tutorials" section that mixes the core 652-line SO-101 walkthrough (`il_robots.mdx`) with RL (950-line `hilserl.mdx`), PEFT, multi-GPU, and rename-map content — four different audiences in one flat list.
2. **Zero decision guidance.** 11 robot pages, 12 policy pages, 10 benchmark pages — all flat alphabetical-ish lists. Nothing answers "which robot should I buy?", "which policy after ACT?", "which benchmark fits my claim?".
3. **The best beginner content isn't on the docs site.** `AGENT_GUIDE.md` (repo root) contains the most complete procedural guidance we have — data-collection golden rules, policy selection by VRAM tier, training-duration heuristics, eval success-rate targets — and none of it is published.
4. **Structural debt.** 14 orphan `policy_*_README.md` files sit in `docs/source/` unreferenced by the toctree (verbatim hand-copies of `src/lerobot/policies/*/README.md`); 6 of 17 CLI commands are completely undocumented; 1 of 15 teleoperators has a doc page; there is no troubleshooting page and no `_redirects.yml`.
5. **Pages do too many jobs.** Tutorial, how-to, and reference content is interleaved (e.g., `lerobot-dataset-v3.mdx` is simultaneously a format spec, a Python API guide, and a migration guide), which makes every page long and none of them skimmable.

**The proposal, in five points:**

1. **A numbered "Learn: your first robot" course** (FastAPI-style): 11 small CLI-only pages taking an SO-101 owner from install to a trained, evaluated ACT policy. Each page ends with a checkpoint and a "Next" link. A one-page no-hardware quickstart runs in parallel.
2. **A sidebar that reads top-to-bottom as increasing expertise**, organized into 17 sections: Get started → task-oriented how-tos (Collect data / Train / Deploy & evaluate) → hardware catalogs → policy catalogs → concepts → RL → extending → reference → community.
3. **Every catalog section opens with a comparison/decision page**: Choosing a robot, Teleoperator overview, Choosing a policy, Reward models overview, Choosing a benchmark.
4. **Publish the missing content**: AGENT_GUIDE.md's tips become real docs pages; a full CLI reference covers all 17 commands; a symptom-organized troubleshooting page aggregates fixes currently buried across five pages.
5. **Minimal URL breakage**: existing slugs are kept wherever possible. Only **2 redirects** are needed (`cheat-sheet → cli_reference`, `il_robots → learn/index`); everything else is a toctree reposition.

**Effort estimate:** ~4–5 person-weeks across 5 phases; phases 3 and 5 are parallelizable and community-friendly (see §9).

---

## 2. Current-state audit

### 2.1 By the numbers

| Metric | Value |
|---|---|
| Files in `docs/source/` (excl. `_toctree.yml`) | 90 |
| Pages in the toctree | 76 |
| Orphan files (on disk, not in toctree) | 14 (`policy_*_README.md`) |
| Top-level sidebar sections | 15, all flat (no nesting) |
| Longest pages | `hilserl.mdx` ~950 lines · `il_robots.mdx` ~650 · `so100.mdx` ~640 |
| CLI commands in `pyproject.toml [project.scripts]` | 17 — **6 with zero doc presence** (`lerobot-find-cameras`, `lerobot-setup-can`, `lerobot-find-joint-limits`, `lerobot-train-tokenizer`, `lerobot-imgtransform-viz`, `lerobot-info`) |
| Teleoperators in `src/lerobot/teleoperators/` | 15 — **1 documented** (`phone_teleop.mdx`); keyboard, gamepad, leader arms covered only implicitly inside robot/tutorial pages |
| Camera backends | 4 — `zmq` undocumented, `reachy2_camera` implicit |
| Policies in `src/lerobot/policies/` | 19 — `diffusion`, `tdmpc`, `vqbet` have orphan READMEs but **no doc page**; `sac`, `gaussian_actor` undocumented |
| Redirect infrastructure | None (`_redirects.yml` does not exist) |

### 2.2 Prioritized UX problems

**Critical (blocks the main funnel):**

| # | Problem | Evidence |
|---|---|---|
| C1 | No linear beginner path: install → ??? | "Get started" = `index.mdx` (brand copy) + `installation.mdx` + `cheat-sheet.mdx`. The actual path lives inside `il_robots.mdx`, filed under "Tutorials" with 9 unrelated pages. |
| C2 | No "which robot do I buy?" guidance | 11 robot pages, no comparison. SO-101 vs SO-100 relationship is never stated; SO-100's page is *longer* than SO-101's. |
| C3 | `il_robots.mdx` monolith | One 652-line page covers calibrate + teleoperate + cameras + record + train + evaluate, doubling every step in CLI *and* Python, all SO-101-specific despite the generic title. |
| C4 | Best procedural content unpublished | `AGENT_GUIDE.md` §5 (data tips), §6 (policy choice), §7 (training duration), §8 (eval targets) exist only in the repo root. |
| C5 | No hardware-free on-ramp | No "train on a Hub dataset in 15 minutes" page; `notebooks.mdx` is 30 lines of links. |

**High (breaks common tasks):**

| # | Problem | Evidence |
|---|---|---|
| H1 | Training guidance scattered | Spread across `act.mdx`, `hardware_guide.mdx`, `multi_gpu_training.mdx`, `il_robots.mdx` — no single "how to train" page. |
| H2 | No policy comparison/decision page | 12 policy pages; ACT says "recommended first" but nothing maps VRAM/task/data-size → policy. |
| H3 | Recording how-to fused with format spec | `il_robots.mdx` (procedure) vs `lerobot-dataset-v3.mdx` (spec + Python API + migration) — neither is a clean reference or a clean guide. |
| H4 | No benchmark selection guidance | 10 benchmark pages in a flat list; "Adding a New Benchmark" (contributor content) sits *first*. |
| H5 | No troubleshooting page | Fixes are buried in `installation.mdx`, `il_robots.mdx`, `lerobot-dataset-v3.mdx` "Common Issues", and AGENT_GUIDE §5.8. |
| H6 | Teleoperators near-invisible | 15 implementations, 1 page. Keyboard/gamepad (`--teleop.type=keyboard|gamepad`) have no docs at all. |

**Medium / low (debt and polish):**

| # | Problem | Evidence |
|---|---|---|
| M1 | 14 orphan `policy_*_README.md` in `docs/source/` | Verbatim copies of `src/lerobot/policies/*/README.md`; not in toctree, not built; no sync script exists (verified by repo-wide grep) — hand-copied and already drifting. |
| M2 | 6/17 CLI commands undocumented | See §2.1. `cheat-sheet.mdx` covers only the main workflow commands. |
| M3 | Mixed-audience sections | "Tutorials" holds beginner + RL-researcher + contributor content; "Benchmarks" holds user + contributor content. |
| M4 | Naming inconsistencies | "SO-101"/"SO101"/"so101" in prose; `<Tip>` vs `> [!NOTE]` admonitions mixed; hyphen vs underscore slugs (`lerobot-dataset-v3` vs `multi_gpu_training`). |
| M5 | Section opener pages missing | "Robot Processors" starts at "Introduction to Robot Processors" (good) but Robots/Policies/Benchmarks/Teleoperators start with an arbitrary instance page. |

---

## 3. What great docs do (and what we borrow)

**Diátaxis** (diataxis.fr — adopted by Canonical, Cloudflare, Gatsby): documentation serves four distinct needs that should never share a page — *tutorials* (learning by doing), *how-to guides* (task recipes), *reference* (lookup facts), *explanation* (concepts). LeRobot's biggest pages fail precisely because they fuse three of these. → **We borrow:** the four-way separation as the organizing principle of the new sidebar; splits like `lerobot-dataset-v3` (spec) vs `using_lerobot_dataset` (how-to).

**FastAPI**: famous for a numbered, linear tutorial where each chapter is small, runnable, and builds on the previous one; advanced topics live in a separate section so the learn-path is never interrupted. → **We borrow:** the numbered `learn/` course with one outcome per page and a "Next:" footer; advanced content (RL, multi-GPU) moved out of the beginner's way.

**Ultralytics YOLO**: handles a many-models × many-tasks × many-modes matrix with *orthogonal* navigation axes plus comparison pages, instead of forcing one hierarchy. → **We borrow:** Robots, Teleoperators, Policies, and Benchmarks as separate catalog sections, each opening with a comparison table ("Choosing a…" pages).

**HF Transformers** (same doc-builder tooling — everything it does, we can do): card-grid landing page, `<hfoptions>` tabs with **persistent selection** (pick PyTorch once, every code block follows), nested collapsible toctree sections, subdirectory slugs (`model_doc/bert`), `_redirects.yml`. → **We borrow:** card-based `index`, site-wide `<hfoptions id="robot">` / `id="os"` / `id="train_env"` tabs, a collapsible nested course section, and the redirects file.

---

## 4. Proposed information architecture

**17 sections, ~101 toctree pages** (today: 15 sections, 76 pages — growth is almost entirely the 11 deliberately-small course pages; the 14 orphans are absorbed into 2 pages). The sidebar reads top-to-bottom as increasing expertise. Slugs in parentheses; annotations: **[NEW]** to be written · **[KEPT]** same file/slug · **[MOVED]** same file, new section · **[SPLIT from X]** · **[MERGED from X+Y]**.

```
Get started
├─ LeRobot (index) ........................................ [KEPT — rewritten as card-grid landing]
├─ Installation (installation) ............................ [KEPT — full install matrix, reference]
├─ Learn: your first robot — SO-101 course (nested, collapsible)
│  ├─ Welcome: what you'll build (learn/index) ............ [NEW]
│  ├─ 1. Install LeRobot (learn/install) .................. [SPLIT from installation — happy path only]
│  ├─ 2. Assemble your SO-101 (learn/assemble) ............ [SPLIT from so101]
│  ├─ 3. Set up the motors (learn/motors) ................. [SPLIT from so101]
│  ├─ 4. Calibrate (learn/calibrate) ...................... [SPLIT from so101 + il_robots]
│  ├─ 5. Teleoperate your arms (learn/teleoperate) ........ [SPLIT from il_robots]
│  ├─ 6. Connect your cameras (learn/cameras) ............. [SPLIT from cameras + il_robots]
│  ├─ 7. Record a dataset (learn/record) .................. [SPLIT from il_robots + AGENT_GUIDE §5.7]
│  ├─ 8. Train your first policy (learn/train) ............ [SPLIT from il_robots + AGENT_GUIDE §7]
│  ├─ 9. Evaluate your policy (learn/evaluate) ............ [SPLIT from il_robots + AGENT_GUIDE §8]
│  └─ 10. Next steps (learn/next_steps) ................... [NEW]
├─ Quickstart without a robot (quickstart_no_robot) ....... [NEW — AGENT_GUIDE §3 Path B + notebooks]
└─ Troubleshooting (troubleshooting) ...................... [NEW — aggregated from 5 pages + AGENT_GUIDE §5.8]

Collect data
├─ Record a dataset (record_dataset) ...................... [SPLIT from il_robots — robot-agnostic, CLI+Python]
├─ Get high-quality data (data_collection_tips) ........... [NEW — AGENT_GUIDE §5]
├─ Visualize & replay episodes (visualize_replay) ......... [SPLIT from il_robots + using_dataset_tools]
├─ Edit datasets (using_dataset_tools) .................... [KEPT — retitled]
├─ Port large datasets to v3 (porting_datasets_v3) ........ [KEPT]
└─ Add language instructions (language_and_recipes) ....... [KEPT — retitled]

Train policies
├─ Train a policy (train_policy) .......................... [SPLIT from il_robots + AGENT_GUIDE §7; local/Colab/HF-Jobs tabs]
├─ Compute requirements (hardware_guide) .................. [KEPT — retitled]
├─ Multi-GPU training (multi_gpu_training) ................ [MOVED]
├─ Fine-tune with PEFT / LoRA (peft_training) ............. [MOVED]
├─ PyTorch accelerators (torch_accelerators) .............. [KEPT]
└─ Rename map & empty cameras (rename_map) ................ [MOVED]

Deploy & evaluate
├─ Evaluate a policy on your robot (evaluate_policy) ...... [SPLIT from il_robots + AGENT_GUIDE §8]
├─ Deploy a policy — lerobot-rollout (inference) .......... [KEPT]
├─ Async inference (async) ................................ [KEPT]
└─ Real-time chunking — RTC (rtc) ......................... [KEPT — absorbs policy_rtc_README.md]

Robots
├─ Choosing a robot (choose_a_robot) ...................... [NEW — comparison table + decision guidance]
├─ SO-101 (so101) ......................................... [SPLIT — assembly/motors/calibration → course; page becomes spec + sourcing + links]
├─ SO-100 — previous generation (so100) ................... [KEPT + legacy banner]
├─ Koch v1.1 (koch) · LeKiwi (lekiwi) · Hope Jr (hope_jr) · Reachy 2 (reachy2)
│  Unitree G1 (unitree_g1) · Earth Rover Mini (earthrover_mini_plus)
│  OMX (omx) · OpenArm (openarm) · reBot B601-DM (rebot_b601) ... [all KEPT, normalized to robot template]

Teleoperators
├─ Teleoperator overview (teleoperators_overview) ......... [NEW — pairing matrix for all 15; keyboard/gamepad usage]
└─ Phone teleoperation (phone_teleop) ..................... [KEPT]

Cameras & motors
├─ Cameras (cameras) ...................................... [KEPT — expanded: ZMQ + Reachy 2 backends]
├─ Feetech motors & firmware (feetech) .................... [MOVED]
└─ Damiao motors & CAN bus (damiao) ....................... [MOVED — documents lerobot-setup-can]

Policies
├─ Choosing a policy (choose_a_policy) .................... [NEW — AGENT_GUIDE §6: VRAM tiers, decision rules]
├─ ACT (act) .............................................. [KEPT — absorbs policy_act_README.md]
├─ Diffusion Policy (diffusion) ........................... [NEW — from policy_diffusion_README.md; no page exists today]
├─ SmolVLA (smolvla) · π₀ (pi0) · π₀-FAST (pi0fast) · π₀.₅ (pi05)
│  GR00T N1.5 (groot) · MolmoAct2 (molmoact2) · VLA-JEPA (vla_jepa) · EO-1 (eo1)
│  X-VLA (xvla) · Multitask DiT (multi_task_dit) · WALL-OSS (walloss) ... [all KEPT, absorb matching READMEs, normalized to policy template]
└─ Legacy policies — VQ-BeT, TDMPC (legacy_policies) ...... [NEW — MERGED from policy_vqbet_README + policy_tdmpc_README]

Reward models
├─ Reward models overview (reward_models_overview) ........ [NEW]
└─ SARM (sarm) · ROBOMETER (robometer) · TOPReward (topreward) ... [KEPT]

Datasets in depth
├─ LeRobotDataset format — v3 spec (lerobot-dataset-v3) ... [SPLIT — keeps slug; format/layout/migration only]
├─ Load & stream datasets in Python (using_lerobot_dataset) [SPLIT from lerobot-dataset-v3 — Python API, streaming, transforms]
├─ Tool-calling columns (tools) ........................... [KEPT — retitled]
├─ Video encoding parameters (video_encoding_parameters) .. [KEPT]
└─ Streaming video encoding (streaming_video_encoding) .... [KEPT]

Simulation
├─ Environments from the Hub (envhub) ..................... [KEPT — section opener]
├─ LeIsaac: control & train in Isaac Sim (envhub_leisaac) . [KEPT]
└─ NVIDIA IsaacLab Arena (envhub_isaaclab_arena) .......... [MOVED from Benchmarks]

Benchmarks
├─ Choosing a benchmark (choose_a_benchmark) .............. [NEW — comparison + match-benchmark-to-claim guidance]
└─ LIBERO (libero) · LIBERO-plus (libero_plus) · Meta-World (metaworld)
   RoboTwin 2.0 (robotwin) · RoboCasa365 (robocasa) · RoboCerebra (robocerebra)
   RoboMME (robomme) · VLABench (vlabench) ................ [all KEPT]

Processors
├─ What are processors? (introduction_processors) ......... [KEPT — entry-level rewrite + diagram]
├─ Processors for robots & teleops (processors_robots_teleop) [KEPT]
├─ Environment processors (env_processor) ................. [KEPT]
├─ Action representations (action_representations) ........ [KEPT]
└─ Debug a processor pipeline (debug_processor_pipeline) .. [KEPT]

Reinforcement learning
├─ Train with RL on a real robot — HIL-SERL (hilserl) ..... [MOVED — split candidate, see §10 Q7]
├─ Train RL in simulation (hilserl_sim) ................... [MOVED]
└─ Human-in-the-loop data collection (hil_data_collection)  [MOVED]

Extending LeRobot
├─ Add a policy (bring_your_own_policies) ................. [MOVED]
├─ Add a robot (integrate_hardware) ....................... [MOVED — retitled]
├─ Write your own processor (implement_your_own_processor)  [MOVED]
└─ Add a benchmark (adding_benchmarks) .................... [MOVED]

Reference & resources
├─ CLI reference (cli_reference) .......................... [MERGED from cheat-sheet + NEW content for 6 undocumented commands]
├─ LeLab: browser GUI (lelab) ............................. [MOVED — placement is open question Q5]
├─ Notebooks (notebooks) .................................. [MOVED]
└─ Backward compatibility (backwardcomp) .................. [MOVED]

Community
└─ Contribute to LeRobot (contributing) ................... [KEPT]
```

**`_redirects.yml`** (complete file — only two entries):

```yaml
cheat-sheet: cli_reference
il_robots: learn/index
```

Everything else keeps its slug; toctree moves and retitles don't change URLs in doc-builder.

---

## 5. The beginner course, page by page

**Template for every course page:** goal sentence → "You need" list → numbered steps, each ending with a verifiable checkpoint ("you should see…") → mini-troubleshooting (≤3 most common failures) → "Next:" footer link. **CLI-only — Python appears nowhere in the course.** OS-specific commands use `<hfoptions id="os">`. Budget ≤200 lines per page (assembly exempt: media-heavy).

| # | Slug | Title | Goal — the user can… | Content source | Length | Next |
|---|---|---|---|---|---|---|
| 0 | `learn/index` | Welcome: what you'll build | See the outcome (a trained ACT policy moving a real SO-101), the shopping list (~$200 kit, 2 cameras, GPU or Colab), time budget (~1 weekend) | NEW; framing from AGENT_GUIDE §2–3 | ~120 | 1 |
| 1 | `learn/install` | Install LeRobot | Working env via one happy path (`pip install 'lerobot[feetech]'`), verified with `lerobot-info` | `installation.mdx` steps 1–3, trimmed to one path | ~100 | 2 |
| 2 | `learn/assemble` | Assemble your SO-101 | Source/print parts, assemble leader + follower | `so101.mdx` sourcing + assembly sections (videos carry the load) | ~400 | 3 |
| 3 | `learn/motors` | Set up the motors | Find ports (`lerobot-find-port`), set IDs (`lerobot-setup-motors`) for both arms | `so101.mdx` motor config + `cheat-sheet.mdx` | ~150 | 4 |
| 4 | `learn/calibrate` | Calibrate | Run `lerobot-calibrate` on both arms; know where calibration files live | `so101.mdx` + `il_robots.mdx` calibration sections | ~120 | 5 |
| 5 | `learn/teleoperate` | Teleoperate your arms | Run `lerobot-teleoperate` — the first "wow" moment, deliberately **before** cameras | `il_robots.mdx` teleop (CLI only) | ~80 | 6 |
| 6 | `learn/cameras` | Connect your cameras | Detect with `lerobot-find-cameras`, position wrist + front cams, teleop with camera view | `cameras.mdx` + `il_robots.mdx` + AGENT_GUIDE §5.1 placement tips | ~120 | 7 |
| 7 | `learn/record` | Record a dataset | Record 50 episodes of one task with `lerobot-record`; keyboard controls, resume, Hub push | `il_robots.mdx` record + AGENT_GUIDE §5.7 defaults ("50 episodes, one task, fixed camera") + §5.2 | ~180 | 8 |
| 8 | `learn/train` | Train your first policy (ACT) | Launch `lerobot-train` with sane ACT defaults; know how long to wait and when to stop. Tabs: local GPU / Colab / HF Jobs (`<hfoptions id="train_env">`) | `il_robots.mdx` train + AGENT_GUIDE §7.1, §7.3, §7.7 | ~150 | 9 |
| 9 | `learn/evaluate` | Evaluate your policy | Run the policy, measure success over 10 trials, know what's "good" | `il_robots.mdx` eval + AGENT_GUIDE §8.1, §8.3, §5.8 | ~120 | 10 |
| 10 | `learn/next_steps` | Next steps | Pick a direction: better data → `data_collection_tips`; language/multi-task → `smolvla`; mobile → `lekiwi`; RL → `hilserl`; sim → `envhub`; Discord | NEW; card grid | ~60 | — |

**Parallel track — `quickstart_no_robot`:** one page mirroring steps 7–9 without hardware: pick a Hub dataset → `lerobot-dataset-viz` → train ACT (Colab button + local command) → evaluate in sim. Ends with "got a robot? start the course." Source: AGENT_GUIDE §3 Path B + `notebooks.mdx`. (Blessed dataset/env to be decided — §10 Q8.)

---

## 6. New pages — content briefs

**Hub / decision pages:**

- **`choose_a_robot`** — Comparison table of all 11 robots: photo, type (arm / mobile / humanoid / hand), DoF, motor family (Feetech / Dynamixel / Damiao), approx. price, sourcing (kit / 3D-print / buy), teleop options, docs maturity. Below: short decision paragraphs ("first robot → SO-101", "mobile manipulation → LeKiwi", "research platform → Reachy 2 / G1"). Ends with a card into the course.
- **`teleoperators_overview`** — Pairing matrix: all 15 teleoperators × compatible robots, with the exact `--teleop.type=` value for each. Inline usage sections for **keyboard** and **gamepad** (currently undocumented); leader arms link to their robot pages; phone links to `phone_teleop`.
- **`choose_a_policy`** — Table from AGENT_GUIDE §6.1: policy, params, min VRAM tier, single/multi-task, language-conditioned, inference speed, pretrained checkpoints, dataset-size sweet spot. Decision rules from §6.2 ("first policy → ACT; language + 16 GB → SmolVLA; ≥40 GB → π₀ / GR00T"). Legacy policies in a collapsed footnote.
- **`reward_models_overview`** — Half page: what reward models are for (success classification for RL, episode filtering, eval scoring), where they plug into HIL-SERL, 3-row table (SARM / ROBOMETER / TOPReward: input modality, output, use case).
- **`choose_a_benchmark`** — Table: benchmark, sim engine, # tasks, focus (long-horizon / language / generalization / bimanual), policies with reported results, GPU needs, Dockerfile availability (AGENT_GUIDE §8.2b). Guidance: match the benchmark to your policy's claim.

**Getting started:**

- **`index` (rewrite)** — Card grid replacing prose: "Build your first robot (course)" / "Quickstart without a robot" / "Choose a robot" / "Choose a policy" / "Browse datasets & models on the Hub" / "Join Discord". Keep the one-paragraph mission statement.
- **`troubleshooting`** — Symptom-organized, one H2 per failure area: installation (ffmpeg, CUDA) · ports & permissions (find-port fails, udev) · motors (wrong ID, no torque, LED codes from AGENT_GUIDE) · calibration drift · cameras (fps, USB bandwidth) · recording (crashes, resume) · training (flat loss, OOM → compute guide) · policy acts erratically (AGENT_GUIDE §5.8 signal table). Each entry: symptom → cause → fix command.

**How-to consolidations (Diátaxis splits):**

- **`record_dataset`** — Robot-agnostic deep version of course step 7: full `lerobot-record` flag reference, `<hfoptions id="robot">` per-robot commands, Python API in its own H2, Hub upload.
- **`data_collection_tips`** — AGENT_GUIDE §5 nearly verbatim: ergonomics, practice runs, consistency, the "start small" golden rule, troubleshooting signals.
- **`visualize_replay`** — `lerobot-dataset-viz`, the online Hub visualizer, `lerobot-replay`.
- **`train_policy`** — One canonical training how-to: `lerobot-train` anatomy, steps/batch/LR guidance (AGENT_GUIDE §7), resume, checkpoints, W&B, local/Colab/HF-Jobs tabs. Links out to multi-GPU, PEFT, compute guide.
- **`evaluate_policy`** — Real-robot eval protocol (n trials, success criteria, §8.3 targets), `lerobot-eval` for sim, comparing checkpoints.
- **`using_lerobot_dataset`** — Python API split out of `lerobot-dataset-v3`: load from Hub, random access, `delta_timestamps`, DataLoader, streaming, image transforms (+ `lerobot-imgtransform-viz`).

**Reference:**

- **`cli_reference`** — Replaces cheat-sheet. All 17 commands, one H2 each, grouped by workflow: *Setup* (find-port, setup-motors, setup-can, calibrate, find-cameras, info) · *Data* (record, replay, dataset-viz, edit-dataset, imgtransform-viz, find-joint-limits) · *Train* (train, train-tokenizer) · *Deploy* (eval, rollout, teleoperate). Each: one-line purpose, copy-paste SO-101 example, link to the relevant guide.
- **`diffusion`** — Standard policy-template page built from `policy_diffusion_README.md` (the only major policy with no page today).
- **`legacy_policies`** — VQ-BeT + TDMPC on one page with a status banner ("maintained for reproducibility, not recommended for new projects"), minimal train/eval commands, paper links.

---

## 7. Migration table

**Redirects needed: 2.** Orphan files aren't in the toctree (never built/served), so deleting them breaks no URLs. Toctree moves and retitles with unchanged slugs need no redirects.

### 7.1 Current toctree pages (76)

| Old slug | Disposition | New location | Redirect |
|---|---|---|---|
| `index` | rewrite (card grid) | Get started | no |
| `installation` | keep (reference matrix); happy path copied to `learn/install` | Get started | no |
| `cheat-sheet` | **merge** into `cli_reference` | Reference & resources | **yes** |
| `il_robots` | **split 8 ways** → `learn/{calibrate,teleoperate,cameras,record,train,evaluate}`, `record_dataset`, `visualize_replay`, `train_policy`, `evaluate_policy` | — | **yes → `learn/index`** |
| `lelab` | move | Reference & resources | no |
| `bring_your_own_policies` | move, retitle "Add a policy" | Extending | no |
| `integrate_hardware` | move, retitle "Add a robot" | Extending | no |
| `hilserl`, `hilserl_sim`, `hil_data_collection` | move | Reinforcement learning | no |
| `multi_gpu_training`, `peft_training`, `torch_accelerators`, `rename_map` | move | Train policies | no |
| `hardware_guide` | move, retitle "Compute requirements" | Train policies | no |
| `lerobot-dataset-v3` | **split**: keeps slug as format spec; Python usage → `using_lerobot_dataset` | Datasets in depth | no |
| `porting_datasets_v3`, `language_and_recipes` | move | Collect data | no |
| `using_dataset_tools` | move, retitle "Edit datasets"; viz section → `visualize_replay` | Collect data | no |
| `tools` | move, retitle "Tool-calling columns" | Datasets in depth | no |
| `video_encoding_parameters`, `streaming_video_encoding` | move | Datasets in depth | no |
| `act`, `smolvla`, `pi0`, `pi0fast`, `pi05`, `molmoact2`, `vla_jepa`, `eo1`, `groot`, `xvla`, `multi_task_dit`, `walloss` | keep; normalize to policy template; absorb matching orphan READMEs | Policies (after `choose_a_policy`) | no |
| `sarm`, `robometer`, `topreward` | keep | Reward models (after overview) | no |
| `inference` | keep, retitle "Deploy a policy (lerobot-rollout)" | Deploy & evaluate | no |
| `async` | keep | Deploy & evaluate | no |
| `rtc` | keep; absorb `policy_rtc_README.md` | Deploy & evaluate | no |
| `envhub`, `envhub_leisaac` | keep | Simulation | no |
| `envhub_isaaclab_arena` | move | Simulation | no |
| `adding_benchmarks` | move, retitle "Add a benchmark" | Extending | no |
| `libero`, `libero_plus`, `metaworld`, `robotwin`, `robocasa`, `robocerebra`, `robomme`, `vlabench` | keep | Benchmarks (after `choose_a_benchmark`) | no |
| `introduction_processors` | keep; entry-level rewrite | Processors | no |
| `processors_robots_teleop`, `env_processor`, `action_representations`, `debug_processor_pipeline` | keep | Processors | no |
| `implement_your_own_processor` | move | Extending | no |
| `so101` | **split**: assembly → `learn/assemble`, motors → `learn/motors`, calibration → `learn/calibrate`; page becomes spec/sourcing/overview linking into the course | Robots | no (slug unchanged) |
| `so100` | keep + legacy banner ("SO-101 is the current generation") | Robots | no |
| `koch`, `lekiwi`, `hope_jr`, `reachy2`, `unitree_g1`, `earthrover_mini_plus`, `omx`, `openarm`, `rebot_b601` | keep; normalize to robot template | Robots (after `choose_a_robot`) | no |
| `phone_teleop` | keep | Teleoperators (after overview) | no |
| `cameras` | keep; expand (ZMQ + Reachy 2 backends) | Cameras & motors | no |
| `feetech`, `damiao` | move; document `lerobot-setup-can` in damiao | Cameras & motors | no |
| `notebooks`, `backwardcomp` | move | Reference & resources | no |
| `contributing` | keep | Community | no |

### 7.2 Orphan files (14 — delete from `docs/source/`, no redirects needed)

Canonical copies remain in `src/lerobot/policies/*/README.md`. Verified: the docs/source copies are byte-identical hand-copies with **no sync script anywhere in the repo** — deletion is safe once unique content is folded in.

| Orphan file | Disposition |
|---|---|
| `policy_act_README.md`, `policy_groot_README.md`, `policy_molmoact2_README.md`, `policy_multi_task_dit_README.md`, `policy_pi0_README.md`, `policy_pi05_README.md`, `policy_sarm_README.md`, `policy_smolvla_README.md`, `policy_vla_jepa_README.md`, `policy_walloss_README.md`, `policy_rtc_README.md` | Delete; fold any unique content into the corresponding `.mdx` page |
| `policy_diffusion_README.md` | Delete; content seeds the **new** `diffusion.mdx` |
| `policy_tdmpc_README.md`, `policy_vqbet_README.md` | Delete; content seeds the **new** `legacy_policies.mdx` |

### 7.3 New files (27)

`learn/index`, `learn/install`, `learn/assemble`, `learn/motors`, `learn/calibrate`, `learn/teleoperate`, `learn/cameras`, `learn/record`, `learn/train`, `learn/evaluate`, `learn/next_steps`, `quickstart_no_robot`, `troubleshooting`, `record_dataset`, `data_collection_tips`, `visualize_replay`, `train_policy`, `evaluate_policy`, `using_lerobot_dataset`, `choose_a_robot`, `teleoperators_overview`, `choose_a_policy`, `diffusion`, `legacy_policies`, `reward_models_overview`, `choose_a_benchmark`, `cli_reference` — plus `_redirects.yml`.

---

## 8. Conventions & style guide

**Naming**
- Product names in prose: **SO-101**, SO-100, LeKiwi, π₀ (display) / `pi0` (slug/code), HIL-SERL, LeRobotDataset. Never "so101"/"SO101" in prose.
- Slugs: lowercase with underscores for new files (`choose_a_robot`); course under `learn/`. Existing hyphenated slugs (`lerobot-dataset-v3`, `cheat-sheet`) are grandfathered, not propagated.
- Titles: sentence case. How-to titles start with a verb ("Record a dataset"); reference titles are nouns ("CLI reference"); course titles are numbered ("3. Set up the motors").
- Placeholders in commands: consistent `<angle_brackets>` (e.g. `--robot.port=<your_port>`).

**Components**
- `<hfoptions>` with **fixed, site-wide ids** so a choice persists across pages: `id="robot"` · `id="os"` · `id="install"` · `id="train_env"` (local/colab/jobs). Tabs only for structurally parallel content; never hide unique content inside a tab.
- `<Tip>` for actionable advice; `<Tip warning={true}>` for anything that can damage hardware, lose data, or cost money. Standardize on `<Tip>` in `.mdx`; `> [!NOTE]` only in plain `.md`. Max ~3 tips per page.
- Card grids only on `index`, `learn/next_steps`, and section overview pages.
- Every assembly/calibration video must have the same steps in text below it (accessibility + searchability).

**Page templates & length budgets**

| Type | Skeleton | Budget |
|---|---|---|
| Course page | goal → "you need" → numbered steps w/ checkpoints → mini-troubleshooting → Next | ≤200 lines (assembly exempt) |
| How-to | problem statement → minimal working command → variations → edge cases → related links | ≤300 |
| Robot page | hero photo → spec table → buy/build → calibration quirks → compatible teleops → known issues | ≤400 |
| Policy page | summary card (params, VRAM, license, paper, checkpoints) → when to use → install → train → finetune → results → citation | ≤250 |
| Concept page | what & why in 3 sentences → diagram → details → pointers to how-tos | ≤400 |

Hard cap **500 lines** for any page → must split (`il_robots` at ~650 and `hilserl` at ~950 are the cautionary examples).

---

## 9. Phased rollout plan

| Phase | Scope | PRs | Effort | Parallel? |
|---|---|---|---|---|
| **0. Sign-off** | This proposal reviewed; open questions (§10) decided | — | ~0.5 wk elapsed | — |
| **1. Skeleton & cleanup** | New `_toctree.yml` (all KEPT/MOVED pages in final positions), `_redirects.yml`, delete 14 orphans, retitles, SO-100 legacy banner | 1 mechanical PR (no content changes — easy review) | 1–2 days | no |
| **2. Beginner course** | 11 `learn/` pages, `so101` + `il_robots` splits, `quickstart_no_robot`, `troubleshooting`; `il_robots` redirect ships here | 3 PRs (steps 0–4 / 5–10 / quickstart+troubleshooting) | 5–8 days | partially — single author for voice; troubleshooting separable |
| **3. Hub / decision pages** | `choose_a_robot`, `choose_a_policy`, `choose_a_benchmark`, `teleoperators_overview`, `reward_models_overview`, `index` rewrite | 6 independent PRs | 4–6 days | **fully** — good first issues per area owner |
| **4. How-to consolidation** | `record_dataset`, `data_collection_tips`, `visualize_replay`, `train_policy`, `evaluate_policy`, dataset split, cheat-sheet → `cli_reference` (+ its redirect) | 3–4 PRs (data / train / deploy / datasets) | 5–7 days | parallel across clusters |
| **5. Reference fill-in** | `diffusion`, `legacy_policies`, ZMQ camera docs, 6 undocumented CLI commands, keyboard/gamepad teleop content, policy/robot pages normalized to templates | many small PRs | 4–6 days | fully — community-friendly |

Total ≈ **4–5 person-weeks**. Phases 2 and 3 can run concurrently after Phase 1 merges. Each URL-breaking change ships in the same PR as its redirect.

---

## 10. Open questions for maintainers

1. **Legacy policies (TDMPC, VQ-BeT):** combined `legacy_policies` page (proposed), full individual pages, or src READMEs only?
2. **Orphan READMEs:** verified byte-identical hand-copies of `src/lerobot/policies/*/README.md` with no sync script — OK to delete from `docs/source/` in Phase 1? Which copy is canonical going forward (proposed: src)?
3. **AGENT_GUIDE.md future:** large parts become docs pages (§5–§8). Slim it to a pointer file for AI agents (avoids divergence), or keep self-contained and accept dual maintenance?
4. **Course URL namespace:** `learn/` (proposed) vs `getting_started/` vs `tutorial/` — permanent once shipped, decide before Phase 2.
5. **LeLab status:** first-party and maintained? If yes, add a "prefer a GUI?" callout inside the course (teleop/record steps); if experimental, keep in Reference & resources as proposed.
6. **SO-100 messaging:** proposed banner "SO-101 is the current generation" + full docs retained. Strong enough, or formally deprecate?
7. **HIL-SERL split:** `hilserl.mdx` is ~950 lines. Split into "Setup & demonstrations" + "Reward classifier & training" in a later phase, or keep monolithic for its expert audience?
8. **Blessed no-hardware quickstart:** PushT, a Hub SO-101 dataset, gym-hil, or an EnvHub env? Determines `quickstart_no_robot` content.
9. **`<hfoptions>` id taxonomy:** agree on the canonical ids (`robot`, `os`, `install`, `train_env`) — they persist site-wide once shipped.
10. **Reward models placement:** standalone section after Policies (proposed) vs nested under Reinforcement learning — depends on whether SARM/ROBOMETER/TOPReward are positioned as general eval tools or RL components.

---

## Appendix: sources & verification

- Current nav: `docs/source/_toctree.yml` (76 pages, 15 sections — verified June 2026).
- File census: 90 content files in `docs/source/`; orphans confirmed by toctree diff; orphan↔src byte-identity confirmed by `diff`; absence of a sync script confirmed by repo-wide grep.
- CLI commands: `pyproject.toml [project.scripts]` (17 entries).
- AGENT_GUIDE.md section references (§5 data tips, §6 policy choice, §7 training duration, §8 evaluation) verified against its headings.
- Patterns referenced: Diátaxis (diataxis.fr) · FastAPI docs · Ultralytics YOLO docs · HF Transformers docs (doc-builder feature ceiling).
