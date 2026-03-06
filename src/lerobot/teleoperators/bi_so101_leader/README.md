# BiSO-101 Leader Teleoperator

```
lerobot-teleoperate \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras='{
        left: {"type": "opencv", "index_or_path": 14, "width": 640, "height": 480, "fps": 30},
        right: {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
        l_side: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30},
        r_side: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}
    }' \
    --teleop.type=bi_so101_leader \
    --teleop.left_arm_port=/dev/ttyACM3 \
    --teleop.right_arm_port=/dev/ttyACM2 \
    --teleop.id=leader \
    --display_data=true
```

---
