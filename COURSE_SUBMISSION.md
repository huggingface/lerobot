# Submitting this project as its own GitHub repository

This folder is a **standalone Git clone** of the agentic-manipulation work (branch `feature/agentic-manipulate` from your fork), with history squashed to a **shallow** latest snapshot (`git clone --depth 1`).

## 1. Create a new empty repository on GitHub

1. Open [github.com/new](https://github.com/new).
2. Choose a name (e.g. `agentic-robot-manipulation` or `so101-vlm-agent`).
3. Leave **no** README, `.gitignore`, or license (the clone already has files).
4. Create the repository.

## 2. Point this clone at the new repo and push

Replace `YOUR_USER` and `YOUR_REPO` with your GitHub username and the new repo name.

```bash
cd /Users/armin/Documents/agentic-manipulation-submission
git remote set-url origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

SSH:

```bash
git remote set-url origin git@github.com:YOUR_USER/YOUR_REPO.git
git push -u origin main
```

After this, **submit the new repo URL** to your course (e.g. `https://github.com/YOUR_USER/YOUR_REPO`).

## 3. What to highlight in your write-up

| Topic | Location |
|-------|----------|
| Agent loop overview | `src/lerobot/agent/README.md` |
| Perception pipeline | `src/lerobot/perception/PIPELINE.md` |
| Main CLI & flags | `src/lerobot/scripts/lerobot_agentic_manipulate.py` (module docstring) |

Entry point after install: `lerobot-agentic-manipulate --help`.

## Note

This tree is still **LeRobot-based** (fork-style layout). If your instructor wants a **minimal** repo with only your files, say so and trim to a smaller tree in a separate folder.
