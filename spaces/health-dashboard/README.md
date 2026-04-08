---
title: LeRobot Health Dashboard
emoji: 🤖
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Live CI health for the LeRobot main branch
---

# LeRobot Health Dashboard

Internal dashboard for monitoring the health of the `main` branch — benchmark smoke-test
success rates, CI job durations, and latest rollout videos, all pulled live from the
GitHub Actions API.

## Required secret

Add `GITHUB_RO_TOKEN` in the Space settings with a fine-grained GitHub token scoped to:

- **Repository**: `huggingface/lerobot`
- **Permissions**: `Actions` → Read-only, `Metadata` → Read-only

The token is never exposed in the UI or logs.
