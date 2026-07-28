# LeRobot Docs Audit

**Status:** Fresh baseline for a new docs redesign effort · **Date:** 2026-07-27
**Supersedes:** the June 2026 `DOCS_REDESIGN.md` proposal (lived on the unmerged, now-stale `docs/complete-docs-redesign` branch — abandoned; that branch is ~67k lines behind `main` and should not be resurrected). This document re-audits `docs/source/` from scratch against current `main`, since the docs tree grew substantially in the ~6 weeks between the two audits (many new policies, robots, and benchmarks landed).

---

## Snapshot (current state, in numbers)

- **IA**: 15 top-level `_toctree.yml` sections, 81 `local:` entries, 0 broken toctree links.
- **Files**: 97 total files in `docs/source/` = 80 real `.mdx` pages + 16 orphaned `policy_*_README.md` stubs (up from 14 in June) + 1 `contributing.md` (a genuine symlink to root `CONTRIBUTING.md`).
- **Growth**: `.mdx` pages grew 49 → 80 in 6 months (+63%, ~5.2 pages/month), arriving in **bursts against a shared, hand-edited `_toctree.yml`** — e.g. 6 benchmark pages landed via 6 separate PRs within 48 hours (Apr 2026); 3 policy pages within 3 days (Jul 2026).
- **CLI docs coverage improved sharply since June**: 17 of 18 `lerobot-*` commands now documented (only `lerobot-info`, a diagnostics helper, is not) — the June baseline was 6/17 undocumented.
- **Decision content**: zero comparison/decision pages exist in `docs/source/` for Policies (15 options), Robots (11), Benchmarks (10), or Teleoperators — this equivalent content already exists, fully written, in root `AGENT_GUIDE.md` §6, but was never ported or linked in.
- **Teleoperators**: dedicated toctree section has only 2 pages vs. 16-17 registered teleoperator types in `src/`, but all types do get at least one doc mention somewhere (usually buried inside a robot's hardware page) — a discoverability gap, not an absence of content.
- **Orphan policy READMEs**: of the 16, 3 (Diffusion, TD-MPC, VQ-BeT) have **zero real published documentation anywhere** — the orphan stub is their only artifact. 13 duplicate an actively-maintained `.mdx` page under a different name. One (MolmoAct2) is proven to have silently rotted: a 39-line pre-refactor stub sits next to a 495-line maintained page that diverged 481 lines ago.
- **Root `README.md` itself has drifted**: its "Supported Hardware" line omits SO-101 (the flagship robot) entirely; its "SoTA Models" table links several policies to thin orphan stubs instead of the richer tutorial pages that exist for them.
- **4 concrete broken/stale copy-paste commands** verified in published pages (`act.mdx`, `il_robots.mdx` ×2 spots, `lerobot-dataset-v3.mdx`, `openarm.mdx` — the last added as recently as Jan 2026, so a fresh miss, not just aging drift).
- **Process/CI**: no CODEOWNERS, no `_redirects.yml`, no doc-consistency or toctree-completeness check anywhere; the docs CI workflow triggers only on `paths: docs/**`, so code-only PRs get zero automated docs signal (empirically responsible for the one real lag found — OMX's docs page, 41 days after its code). Despite this, 18 of 19 spot-checked new integrations (EVO1, FastWAM, LingBot-VA, MolmoAct2, RTC, VLA-JEPA, X-VLA, WallOSS, Damiao, Hope Jr, OpenArm, reBot B601, Reachy 2, Unitree G1, RoboCasa365, VLABench, IsaacLab Arena, etc.) shipped docs in the *same commit* as the feature — organic discipline is currently strong.
- **No content template** exists for policy or robot doc pages: heading sets/length vary 80-528 lines with almost no shared vocabulary; by contrast, benchmarks *do* have a written template (`adding_benchmarks.mdx`) and the resulting 8 pages are visibly consistent — direct proof a template is what produces consistency here.

---

## Strengths

1. **Organic contributor discipline is currently much better than the June audit's numbers implied.** 18/19 spot-checked new policies/robots/benchmarks shipped docs in the same PR as the code; 17/18 CLI commands are documented; all teleoperator types get at least one mention. The underlying practice is healthier than the structural numbers (orphans, thin nav sections) suggest — the real problem is increasingly discoverability and consistency, not absence of effort.
2. **The doc-builder CI pipeline (PR previews, main build, versioned release builds) is solid, standard infrastructure already in place** — nothing bespoke to build or maintain, and it's the same tooling Transformers uses.
3. **A working, provably-effective template pattern already exists for one catalog (benchmarks)** — `adding_benchmarks.mdx`'s checklist + "at a glance" table produces 8 structurally consistent pages. This is the single best evidence in the repo for what fixes the policy/robot template gap, and it needs no new invention, just extension.
4. **Genuine single-source-of-truth patterns already exist and work**: `docs/source/contributing.md` is a real symlink to root `CONTRIBUTING.md` (verified via `ls -la`/`find -type l`). Most policy `src/README.md` files are also symlinked into `docs/source/policy_*_README.md`, eliminating drift at that layer (though the docs-side stub itself remains an unlinked orphan — see weaknesses).
5. **Zero broken toctree links**, and specific pages are genuinely high quality: `cheat-sheet.mdx` (spot-checked line-by-line against source, fully accurate), `bring_your_own_policies.mdx` (clear scope, working PR checklist), `installation.mdx`'s tables/OS tabs, and the SO-100/SO-101 hardware pages.
6. **Root `README.md` already contains a usable taxonomy for policies** (Imitation Learning / RL / VLAs / World Models / Reward Models) that the flat 15-item Policies sidebar never adopted — the raw material for a better IA already exists in-repo.

---

## Weaknesses

### High
- **No comparison/decision-guidance page for any multi-option category** (Policies=15, Robots=11, Benchmarks=10, Teleoperators) — the exact content (decision rules, profiling snapshot) already exists in `AGENT_GUIDE.md` §6 but was never ported into `docs/source`.
- **The landing page (`index.mdx`) provides zero navigation** — 23 lines of marketing copy and a Discord link, no path to installation, cheat-sheet, or audience-specific guidance.
- **Teleoperators' dedicated nav section (2 pages) badly undersells real coverage** (16-17 implementations, all mentioned somewhere) — a new user browsing that sidebar section would wrongly conclude only phone/Isaac teleop exist. Content exists; it's a pure discoverability failure.
- **Multiple concrete broken/stale CLI examples in published, frequently copy-pasted pages**: `act.mdx` tells readers to eval with `lerobot-record` while the shown code uses `lerobot-rollout`; an identical shell-breaking snippet (`\`-terminated comment swallowing a continuation) is duplicated in `il_robots.mdx` and `lerobot-dataset-v3.mdx`; `il_robots.mdx:421` uses the dead `--control.push_to_hub` flag namespace; `openarm.mdx:208-210` (added Jan 2026) gives a `lerobot-record` example with flat pre-draccus flags that don't exist on the current config classes.
- **3 shipping policies (Diffusion, TD-MPC, VQ-BeT) have zero real published documentation** — only a thin, toctree-unreferenced stub exists for each.
- **Root `README.md` has drifted as the project scaled**: its Supported Hardware line omits SO-101 entirely; its SoTA Models table links ACT/Diffusion/VQ-BeT/Multitask-DiT/TDMPC/GR00T/SmolVLA to thin orphan stubs instead of the richer tutorial pages that exist for several of them.
- **No content template for policy/robot doc pages, unlike benchmarks** — heading sets and length vary 80-528 lines with almost no shared vocabulary (`molmoact2.mdx` has no Overview section at all; `smolvla.mdx` has no Overview/Architecture/Citation/License and reads as pure tutorial).
- **Docs CI triggers only on `paths: docs/**`** — code-only PRs get zero automated docs signal; this is the mechanism directly responsible for the one measured lag (OMX's docs page shipped 41 days after its code, in a separate PR).
- **Proof that orphan-stub drift isn't hypothetical**: `policy_molmoact2_README.md` is a stale 39-line pre-refactor snapshot sitting 481 lines behind the maintained 495-line page it once mirrored — nothing caught this silently rotting.

### Medium
- 16 orphaned `policy_*_README.md` files unreferenced by the toctree; 13 are pure duplicate cruft shadowing a maintained same-topic `.mdx` page, creating a false "two files to keep in sync" impression for contributors.
- No `_redirects.yml` anywhere — future cleanup of the orphans/renames has no 404 safety net.
- `installation.mdx` ends on a dangling forward-reference ("follow the link below to use LeRobot with your robot") — no link follows.
- `cheat-sheet.mdx`'s "Policy Types" line is stale (`act, diffusion, smolvla, pi05`) against the current 15-entry Policies section, and lists `diffusion`, which has no working toctree page at all.
- "Tutorials" (10 pages) mixes beginner, contributor, and RL-researcher content in one flat, unlabeled list.
- The EnvHub feature family is split inconsistently across two unrelated sections (`envhub.mdx`/`envhub_leisaac.mdx` under Simulation, `envhub_isaaclab_arena.mdx` under Benchmarks) despite identical framing/opening text.
- Admonition syntax is split GFM `[!NOTE]` vs. doc-builder `<Tip>` across sampled pages; since the site actually builds with `hf-doc-builder`, the majority pattern may not render as a styled callout at all (plausible, not independently render-verified).
- No CODEOWNERS file anywhere — zero designated reviewer routing for `docs/source`.
- `CONTRIBUTING.md` never links the good in-repo extension guides (`adding_benchmarks.mdx`, `bring_your_own_policies.mdx`, `integrate_hardware.mdx`) and states no docs-required policy.
- The "docs required" PR checklist convention is applied inconsistently: Policies and Benchmarks have an explicit required-docs checklist; `integrate_hardware.mdx` (Robots/Teleoperators) has none — a plausible root cause of the Teleoperators gap above.
- Growth repeatedly stacks simultaneous PRs against one shared, hand-edited `_toctree.yml` (6 benchmark PRs in 48h; 3 policy PRs in 3 days) — a merge/oversight risk even though no damage from it was found yet.
- `policy_sarm_README.md` has no source of truth left to sync against at all (SARM moved `policies/`→`rewards/`, no README carried over) — pure abandoned cruft next to the real 593-line `sarm.mdx`.
- No automated check anywhere for toctree completeness or README/mdx sync — the only CI gate is whether the doc-builder build succeeds, which doesn't catch missing coverage.

### Low
- No dedicated "Motors" section despite `motors/` being named as a distinct hardware layer in `CLAUDE.md`; `feetech.mdx`/`damiao.mdx` sit in a catch-all "Resources" section instead.
- "Sensors" is a single-page top-level nav section (`cameras.mdx` only).
- Policies (15), Benchmarks (10), and Robots (11) are flat, ungrouped lists with no internal sub-headings, despite `README.md` already having a usable taxonomy that could be reused.
- "SO-101" vs. "SO101" naming is inconsistent across ~15 files, and even within a single file (`so100.mdx` uses both).
- `hilserl.mdx` (950 lines) and `il_robots.mdx` (638 lines) each mix tutorial, reference, and troubleshooting content in one long page.
- `reachy2_camera` has zero doc mentions anywhere; the `zmq` camera backend is documented only incidentally inside `unitree_g1.mdx` rather than centrally in `cameras.mdx`.
- The PR template's "Documentation updated" line is a self-reported, unenforced checkbox.

---

## Recommendations

### 1. Quick wins (small effort, ship this week, no maintainer proposal needed)
- Fix the 4 verified broken commands: `act.mdx` (`lerobot-record`→`lerobot-rollout`), the duplicated shell-breaking snippet in `il_robots.mdx` + `lerobot-dataset-v3.mdx`, `il_robots.mdx`'s `--control.push_to_hub`→`--dataset.push_to_hub`, and `openarm.mdx`'s invalid flat-flag record example.
- Delete the 3 fully-orphaned policy stubs (Diffusion, TD-MPC, VQ-BeT) and give each a real toctree page by promoting the existing `src/` README content.
- Delete the remaining 13 duplicate orphan stubs and the dead `policy_sarm_README.md`.
- Fix `installation.mdx`'s dangling ending and `cheat-sheet.mdx`'s stale policy-type list.
- Fix root `README.md`: add SO-101 to the Supported Hardware line; repoint the SoTA Models table's links from orphan stubs to the richer existing tutorial pages.
- Move `envhub_isaaclab_arena.mdx` into Simulation (one-line YAML change).
- Add 2-4 orientation sentences + links to the top of `index.mdx` (installation, cheat-sheet, "I have hardware" / "I don't" / "I want to contribute") without touching its marketing framing.
- Publish a single "Choosing a policy" page that ports the already-written decision rules from `AGENT_GUIDE.md` §6 — highest-leverage fix available.
- Start a `docs/source/_redirects.yml` now, before the orphan cleanup above creates the first real dead links.
- No action needed on the `contributing.md` "duplication" claim — verified it's a working symlink, not an anti-pattern.

### 2. Structural / UX changes (need a proposal + maintainer buy-in, phase as separate PRs)
- Split "Tutorials" into an explicit beginner-facing section vs. an "Advanced & Research/Contributor" section (mechanical YAML reorg, no content rewrites).
- Add a "Teleoperators" index page listing all types with one-line descriptions and links to wherever each is actually documented today.
- Re-group the flat Policies/Benchmarks/Robots lists into sub-headings reusing the taxonomy `README.md` already has.
- Treat a dedicated "Motors" section, deeper sub-grouping, and closing the remaining teleoperator/robot coverage gaps as a phased backlog of independently reviewable PRs rather than one restructure.
- Standardize SO-101/SO101 naming and (after confirming actual doc-builder rendering behavior) the admonition syntax, each as one mechanical, low-risk PR.
- Longer-term and biggest-ticket: port more of `AGENT_GUIDE.md`'s procedural content (training duration heuristics, eval targets, data-collection tips) into the published site.

### 3. Maintainability / process changes (prevent debt from reaccumulating at the current growth rate)
- Port `adding_benchmarks.mdx`'s explicit "writing a doc page" checklist/template into `bring_your_own_policies.mdx` and `integrate_hardware.mdx`.
- Add a CODEOWNERS entry for `docs/source/` (and ideally `_toctree.yml` specifically) so docs PRs get routed to a real reviewer.
- Link the extension guides from `CONTRIBUTING.md` and state a docs-required policy there.
- Add a lightweight CI/pre-commit script (not a full doc-builder run) that fails when (a) a `docs/source/*.mdx` file isn't reachable from `_toctree.yml`, or (b) a new `register_subclass` policy/robot/teleoperator/env lands with no corresponding doc file in the same diff.
- Decide the fate of the `policy_*_README.md` symlink convention going forward: it is currently self-perpetuating because `bring_your_own_policies.mdx`'s own checklist instructs new contributors to create the file that ends up orphaned. Either fold citation/paper content into the main `.mdx` tutorial, or wire the stub into the toctree as a linked citation anchor.
- Make the PR template's "Documentation updated" checkbox actionable (e.g., "(N/A if this PR only touches tests/CI/refactors)").
- Defer heavier generated-registry/support-matrix tooling (Transformers/Ultralytics-style single source of truth) until closer to 1.0.

---

## Notes on cross-checking

Five independent audit passes fed this report; two disagreements were resolved during synthesis:
- **Toctree section count** — 15 is correct (independently parsed twice from `_toctree.yml`).
- **`contributing.md`** — it's a working symlink, not a hand-duplicated anti-pattern; one pass's `diff`-based claim didn't survive checking `find -type l`.
- **Orphan-README "sync mechanism"** — mostly fixed at the `src/`↔`docs` layer via symlinks, but that just moved the unresolved drift to the docs-side stub's absence from the toctree, and left old pre-symlink stubs (MolmoAct2) as dead leftovers.
- **Teleoperators coverage** — the nav-section framing ("~2/16") and the "mentioned somewhere" framing are both true; the real gap is discoverability, not content.
