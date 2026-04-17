---
title: LeRobot Benchmark Leaderboard
emoji: 🤖
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Benchmark history for LeRobot policy x benchmark runs
---

# LeRobot Benchmark Leaderboard

This Space reads immutable benchmark rows from a Hugging Face dataset and shows:

- Latest result per policy and benchmark
- Historical trends over time
- Direct links to uploaded eval and config artifacts

## Configuration

Set `BENCHMARK_RESULTS_REPO` in the Space settings if you want to point the UI
at a different public dataset. The default is:

- `lerobot/benchmark-history`
