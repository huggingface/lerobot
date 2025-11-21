lerobot-eval \
  --policy.path="/raid/jade/models/xvla-libero-new_migrated2" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=absolute \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env.episode_length=800 \
  --seed=142
