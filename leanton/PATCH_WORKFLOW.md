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

```bash
SLUG="<descriptive-slug>"   # kebab-case, e.g. "dagger-rtc-fresh-obs-on-resume"
mkdir -p leanton/$SLUG

# Generate diff against main (NOT HEAD~1 — that would include other patches)
git diff main -- <changed_files> > leanton/$SLUG/$SLUG.diff

# Verify clean
git apply --check leanton/$SLUG/$SLUG.diff && echo "✅" || echo "❌"
```

### 3. Document the Patch

Create `leanton/<slug>/README.md`:

```markdown
# <patch title>

**Target:** `<file path in lerobot>`
**Status:** `active`
**GitHub:** `<issue link>` (or "Not filed")

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

### 5. File Upstream (if applicable)

```bash
gh issue create --repo huggingface/lerobot \
  --title "<descriptive title>" \
  --label "bug" --label "policies" \
  --body '<Detailed description with root cause, reproduction, environment, proposed fix>'
```

Record the issue in **both** locations:
- `leanton/<slug>/README.md` — `**GitHub:**` field
- `leanton/UPSTREAM_ISSUES.md` — Open table

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
