HF_USER=Gongsta lerobot-replay \
    --robot.type=bi_koch_follower \
    --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
    --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
    --robot.id=bimanual_follower \
    --dataset.repo_id=${HF_USER}/koch-tshirt-folding-v3 \
    --dataset.episode=2
