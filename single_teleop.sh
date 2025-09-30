lerobot-teleoperate \
    --robot.type=koch_follower \
    --robot.port=$FOLLOWER_LEFT_PORT \
    --robot.id=follower \
    --teleop.type=koch_leader \
    --teleop.port=$LEADER_LEFT_PORT \
    --teleop.id=leader \
    --display_data=true
