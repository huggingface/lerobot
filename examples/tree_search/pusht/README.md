# PushT Tree Search

This directory contains a PushT-specific receding-horizon tree-search evaluator.
It loads policies through the normal LeRobot factories, snapshots the current
PushT simulator state, rolls out hypothetical action chunks, scores the leaves,
restores the real state, and commits the best root action.

Run:

```bash
uv run python examples/tree_search/pusht/search_eval.py \
  --policy.path=aadarshram/act_pusht \
  --policy.device=cuda \
  --policy.use_amp=false \
  --episodes=10 \
  --num-candidates=8 \
  --depth=2 \
  --chunk-size=8
```

Run distinct episodes in parallel without changing the core evaluator:

```bash
uv run python examples/tree_search/pusht/parallel_search_eval.py \
  --policy.path=aadarshram/act_pusht \
  --policy.device=cuda \
  --policy.use_amp=false \
  --episodes=10 \
  --episode-workers=2 \
  --seed=0 \
  --depth=3 \
  --num-candidates=40 \
  --chunk-size=3 \
  --execute-steps=10
```

The wrapper launches `search_eval.py` once per episode, stores each subprocess
under `OUTPUT_DIR/episodes/episode_XXX/`, and writes collated metrics to
`OUTPUT_DIR/eval_info.json`.

Useful knobs:

- `--score-mode=coverage`: use PushT goal coverage as the planner score.
- `--num-candidates`: number of action chunks per expanded node.
- `--noise-std`: candidate noise in PushT action coordinates, where actions are target `(x, y)` positions in `[0, 512]`.
- `--depth`: number of chunk levels to search.
- `--beam-width`: number of nodes kept after each level.
- `--execute-steps`: number of actions from the selected root chunk to commit before replanning.
- `--dump-search-images=true`: save one PNG per planning call under
  `OUTPUT_DIR/search_images/episode_XXX/step_YYYYY.png`. Each image overlays
  candidate action chunks as dots at the agent position after each hypothetical
  action. Blue is the original policy chunk, gray is noisy candidates, and the
  chosen root chunk has a green outline. A matching
  `OUTPUT_DIR/search_images/episode_XXX/step_YYYYY.json` file is also written
  with sorted candidate traces, pixel coordinates, world coordinates, scores,
  and original/selected flags.
- `--dump-frames=true`: save rollout frames as PNGs under
  `OUTPUT_DIR/frames/episode_XXX/frame_YYYYY.png`, next to `videos/`. This is
  independent of `--render-videos`; MP4 generation still uses `--render-videos`.
- `--video_overlay=false`: disable text overlays in rollout MP4s, dumped rollout
  frames, policy trace images, and search images. Dots/paths are still drawn in
  policy/search debug images.
- `--plot_policy_trace`: save raw policy action-trace PNG/JSON pairs under
  `OUTPUT_DIR/policy_frames/episode_XXX/policy_trace_step_YYYYY.png/json` at
  each decision point, even if search later chooses a different chunk. The plot
  shows raw policy action targets and the simulated agent trace for that policy
  chunk.
- `--one_step_further`: before running beam search, simulate the raw policy
  chunk once. If its final PushT coverage does not drop by more than `0.05`,
  skip search and execute the policy chunk directly. If coverage drops by more
  than `0.05`, run search as usual.

Offline trace styling preview:

```bash
uv run python examples/tree_search/pusht/plot_search_trace.py \
  harezmi-extend-dump/pusht/5_eps_viz/n40_c3_d3_e10 \
  --seed=0
```

The plotter chooses a random `search_images/episode_XXX/step_YYYYY.json`, uses
the matching rollout frame from `frames/` when present, and writes a styled PNG
under `RUN_DIR/plot_previews/`. Noisy alternatives use a grayscale gradient, the
original policy chunk uses a blue gradient, and the selected chunk uses the
yellow-to-purple gradient. By default, each candidate trace shows only the first
10 dots for readability; use `--max-dots=0` to draw every stored point.
Later dots fade out by default; use `--no-fade-dots` to keep constant opacity,
or tune the range with `--dot-start-alpha` and `--dot-end-alpha`. Use
`--render-mode=line` to draw only the connected path, `--render-mode=dots` for
dots only, or `--render-mode=both` for both.

The evaluator reports `asr` (Alternative Selection Ratio) in the aggregate
metrics. ASR is the percentage of decision points where the selected root chunk
was a noisy alternative (`selected_candidate_index != 0`) instead of the
original policy chunk.

This is intended for PushT simulation only. Real robots cannot be cloned and
restored without a separate learned or analytic forward model.
