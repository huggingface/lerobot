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
  chosen root chunk has a green outline.
- `--one_step_further`: before running beam search, simulate the raw policy
  chunk once. If its final PushT coverage does not drop by more than `0.05`,
  skip search and execute the policy chunk directly. If coverage drops by more
  than `0.05`, run search as usual.

This is intended for PushT simulation only. Real robots cannot be cloned and
restored without a separate learned or analytic forward model.
