export MUJOCO_GL=egl
python src/lerobot/scripts/eval.py \
    --policy.path=/raid/jade/models/act_aloha \
    --output_dir=outputs/eval/act_aloha_transfer/080000 \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --policy.use_amp=false
