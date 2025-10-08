export MUJOCO_GL=egl
lerobot-eval \
    --policy.path=/home/jade_choghari/robot/robotdev/lerobot/lerobot_act_aloha_sim_transfer_cube_human_migrated \
    --output_dir=outputs/eval/act_aloha_transfer/080000 \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=10 \
    --eval.batch_size=3 \
