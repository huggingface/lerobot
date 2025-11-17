lerobot-eval \
  --policy.path="/raid/jade/models/xvla-libero-og_migrated" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.action_type=abs \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=142
