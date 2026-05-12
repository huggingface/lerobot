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

This is intended for PushT simulation only. Real robots cannot be cloned and
restored without a separate learned or analytic forward model.
