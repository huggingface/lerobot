# Patch Workflow — Standard Operating Procedure

> **SSoT:** This file. Lives in `leanton/` inside [anikita/lerobot](https://github.com/anikita/lerobot), branch `leanton`.
> **Archived patches:** Stored in the vault at `Knowledge_Drafts/LeRobot/Environment_and_SW/leanton/patches/archived/`.

---

## Architecture

```
anikita/lerobot
├── leanton/                    # Patch management (SSoT)
│   ├── README.md               # Active patch inventory
│   ├── PATCH_WORKFLOW.md       # This document
│   ├── UPSTREAM_ISSUES.md      # Open/closed upstream issues
│   └── <slug>/
│       ├── README.md           # Problem, fix, validate
│       └── <slug>.diff         # git diff against main
├── src/lerobot/...             # Patched source (leanton branch only)
└── ...
```

| Branch | Tracks | Patches? | Force-push? |
|:-------|:-------|:---------|:------------|
| `main` | `huggingface/lerobot:main` — clean mirror | No | Never |
| `leanton` | `main` + patches + `leanton/` directory | Yes, applied in source | Yes (rebased) |

## SSoT Split

| Artifact | SSoT | Vault role |
|:---------|:----:|:-----------|
| Patch diffs (`.diff`) | **Fork** `leanton/<slug>/` | None |
| Per-patch README | **Fork** `leanton/<slug>/` | None |
| `PATCH_WORKFLOW.md` | **Fork** `leanton/` | Pointer file (link only) |
| `UPSTREAM_ISSUES.md` | **Fork** `leanton/` | None |
| Patch inventory | **Fork** `leanton/README.md` | None |
| Archived patches | **Vault** `patches/archived/` | SSoT |
| Design docs, runbooks | **Vault** `Knowledge_Drafts/LeRobot/` | SSoT |

---

## Creating a New Patch

### 1. Apply the Fix to Source

```bash
cd ~/lerobot
git checkout leanton
# Edit the target file(s)
# Test on hardware
```

### 2. Create Slug Folder + Diff

> [!CAUTION]
> **MANDATE: Diff MUST be generated against `origin/main` (clean upstream), not against the local `leanton` branch or `HEAD~1`.** A diff against `leanton` omits changes already applied by other patches on the same files — it will fail when applied to upstream `main`. This is non-negotiable. Every `.diff` file must be upstream-ready.

**Why `git diff main -- <files>` is NOT sufficient:** When multiple patches touch the same file, `git diff main` captures ALL patches' cumulative changes, not just yours. Example: our `rollout-policy-revision` patch added `revision: str | None = None` to `PolicyConfig`. If your patch also touches `policies.py`, a `git diff main` will include the `revision` field from the other patch — polluting your diff. Conversely, if you diff against `HEAD~1` (leanton parent), you omit changes from other patches that your code depends on but upstream doesn't have — your diff fails at runtime on upstream.

**Required procedure — isolate your patch on a temp branch from `origin/main`:**

```bash
SLUG="<descriptive-slug>"   # kebab-case, e.g. "dagger-rtc-fresh-obs-on-resume"

# Step A: Record the exact upstream base commit
UPSTREAM_BASE=$(git rev-parse origin/main)
echo "Upstream base: $UPSTREAM_BASE"

# Step B: Create temp branch from clean upstream
git stash  # safety: stash any uncommitted work
git checkout -b tmp-$SLUG origin/main

# Step C: Apply ONLY this patch's source changes to the temp branch
# Option 1: Cherry-pick the commit that has your source changes
git cherry-pick <your-source-commit-hash>
# If cherry-pick conflicts on leanton/ docs, resolve by keeping docs deleted:
#   git rm leanton/$SLUG/...  (or git checkout --theirs / --ours as needed)
#   git cherry-pick --continue

# Option 2 (if cherry-pick is messy): Apply the diff manually
#   git show <your-source-commit> -- src/ | git apply

# Step D: Generate diff against origin/main (CLEAN — only this patch's changes)
git diff origin/main -- src/ > leanton/$SLUG/$SLUG.diff

# Step E: Verify the diff applies cleanly to origin/main
git checkout origin/main -- src/   # reset to upstream state
git apply --check leanton/$SLUG/$SLUG.diff && echo "✅ Upstream-clean" || echo "❌ FAILS — fix before proceeding"

# Step F: Cleanup — return to leanton and restore
git checkout leanton
git branch -D tmp-$SLUG
git stash pop 2>/dev/null  # if you stashed earlier
```

**Document the base commit in the diff header and README:**

The `.diff` file must record the upstream base commit as a comment on line 1:

```diff
# Upstream base: <full commit hash> (huggingface/lerobot:main, <date>)
diff --git a/src/lerobot/...
```

And the README must include:

```markdown
**Diff basis:** `origin/main` @ `<short hash>` (clean upstream).
```

**Verification checklist before committing the `.diff`:**
- [ ] Diff applies cleanly to `origin/main`: `git apply --check <diff>` against upstream files
- [ ] Diff contains ONLY this patch's changes (no leaked changes from other patches)
- [ ] Diff includes any prerequisite fields/config that upstream doesn't have but this patch depends on
- [ ] Upstream base commit hash is documented in the diff and README

### 3. Document the Patch

Create `leanton/<slug>/README.md`:

```markdown
# <patch title>

**Target:** `<file path in lerobot>`
**Status:** `active`
**GitHub:** `<issue link>` (or "Not filed")
**Diff basis:** `origin/main` @ `<short hash>` (clean upstream, <date>)

## What

<One sentence on what the patch does.>

## Why

<Root cause. What breaks without this patch.>

## Validate

**User:** <Steps the user performs on the robot to confirm the patch works.>

**Agent:**
```bash
grep -q "<unique_string>" ~/lerobot/<target_file> && echo "✅" || echo "MISSING"
```
```

**Rules for READMEs:**
- No Obsidian YAML frontmatter — plain markdown, GitHub-friendly
- No vault paths — use fork-relative or standard paths (`~/lerobot/...`)
- No `[[wikilinks]]` — use standard markdown links
- **Diff basis is MANDATORY** — must include the exact `origin/main` commit hash the diff was generated against. This is critical for upstream filing and rebase triage.

### 4. Commit and Push

```bash
cd ~/lerobot
git checkout leanton
git add leanton/$SLUG/ src/lerobot/<changed_files>
git commit -m "[$SLUG] <description>"
git push fork leanton
```

**Commit conventions:**
- Tag commits with slug: `[action-smoothing-ema-clip] Tighten velocity clip`
- One commit per logical change (source + diff + README together)
- Never commit on `main` — it's a clean upstream mirror

### 5. Draft Upstream Issue

Before filing upstream, draft the issue inside the patch folder for review:

```bash
# Create leanton/<slug>/ISSUE.md with:
#   - Title line
#   - Motivation section (why this matters)
#   - Proposed solution (what the patch does, files changed)
#   - Implementation link (branch + diff stats)
#   - Scope / limitations
#   - Testing summary
```

**Review the draft with the user before filing.** Once approved, file the issue:

```bash
gh issue create --repo huggingface/lerobot \
  --title "<title from ISSUE.md>" \
  --body "$(cat leanton/<slug>/ISSUE.md)"
```

### 6. Record the Issue

Track in **all three** locations:
- `leanton/<slug>/ISSUE.md` — canonical draft
- `leanton/<slug>/README.md` — `**GitHub:**` field (update with issue link)
- `leanton/UPSTREAM_ISSUES.md` — Open table

### 7. PR (after issue discussion)

When maintainers accept the approach:
- Create a clean feature branch from `origin/main`: `feat/<slug>`
- Apply the `.diff`: `git apply leanton/<slug>/<slug>.diff`
- Run `pre-commit run -a` and `pytest`
- Follow the [PR template](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
- **Review another contributor's PR first** (community review policy)
- Reference the issue in the PR description

```bash
git checkout -b feat/<slug> origin/main
git apply leanton/<slug>/<slug>.diff
git add src/
git commit -m "<PR title per CONTRIBUTING.md>"
git push fork feat/<slug>
gh pr create --repo huggingface/lerobot --base main --head anikita:feat/<slug>
```

---

## Upgrade Workflow (Upstream Released)

```
1. git checkout main && git pull origin main
2. git checkout leanton && git rebase main
3. For each conflict → triage (see below)
4. For each kept patch → regenerate diff: git diff main -- <files> > leanton/<slug>/<slug>.diff
5. Test on hardware
6. git push fork leanton --force-with-lease
7. Archive any dropped patches in vault
```

### Triage Decision Tree

```
CONFLICT during rebase?
├── GitHub issue closed?           → DROP (archive in vault)
├── Bug pattern gone from main?    → DROP (archive in vault)
├── Target refactored, bug still?  → REBASE (fix conflict, regenerate .diff)
├── Uncertain?                     → SKIP (leave unapplied, flag for manual review)
└── Clean apply?                   → Already applied, just regenerate .diff
```

**Critical:** Step 4 is mandatory even with zero conflicts — upstream changes to the same files shift line numbers, so old diffs become stale. Regenerate every time.

---

## Patch Lifecycle

```
Ideation ──→ Active ──→ Archived (vault)
  │              │              │
  │         Applied in        Trigger:
  │         leanton src       - Upstream merged the fix
  │         .diff + README    - Bug pattern gone from main
  │         in leanton/       - Approach changed
  │                           - Untouched for 6+ months
  │
  └── Folder suffix: _not-implemented
      No .diff. No source changes.
      Lives in fork (visible, not applied).
```

### State Transitions

| From → To | Action |
|:-----------|:-------|
| Ideation → Active | Remove `_not-implemented` suffix, add `.diff` + README, apply to source, commit |
| Active → Archived | Move folder from `leanton/<slug>/` to vault `patches/archived/<slug>/`. Remove from fork. Commit removal. |
| Active → Deleted | Only if patch was applied in error and never worked. Delete folder, revert source. Don't archive. |

### Archive Criteria

A patch moves to archive when:
1. Upstream closed the issue AND the fix is in `main` (verified by code inspection)
2. Rebase conflict reveals the code was refactored AND the bug is gone
3. We stopped using the feature (e.g., switched from diffusion to SmolVLA)
4. 6 months untouched → flagged for review, user decides archive or keep

---

## Colab / External Machine Usage

```bash
git clone https://github.com/anikita/lerobot.git -b leanton
cd lerobot
pip install -e .
# Patches are already in source — no git apply needed
```

---

## Verifying Patch Status

```bash
cd ~/lerobot
git checkout leanton
# Show all patches (changes from main)
git diff main --stat
# Verify all diffs are in sync with source
for d in leanton/*/; do
  slug=$(basename "$d")
  diff="$d/$slug.diff"
  [ -f "$diff" ] || continue
  git apply --check "$diff" && echo "✅ $slug" || echo "❌ $slug — diff is stale, regenerate"
done
```
