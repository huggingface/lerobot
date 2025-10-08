export MUJOCO_GL=egl
lerobot-eval \
    --policy.path=/home/jade_choghari/robot/robotdev/lerobot/lerobot_diffusion_pusht_migrated \
    --output_dir=outputs/eval/diffusion/080000 \
    --env.type=pusht \
    --eval.n_episodes=10 \
    --eval.batch_size=3 \
