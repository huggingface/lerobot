lerobot-eval \
  --policy.path="/raid/jade/models/xvla-lib" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=absolute \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=142

