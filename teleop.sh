#!/bin/bash

# Bimanual teleoperation menu

echo "==================================="
echo "  Bimanual SO-101 Teleoperation"
echo "==================================="
echo ""
echo "1) Basic (no cameras)"
echo "2) With RealSense cameras (RGB only)"
echo "3) With RealSense cameras (RGB + Depth)"
echo "4) Exit"
echo ""
read -p "Select option: " choice

case $choice in
    1)
        echo "Starting bimanual teleoperation..."
        python -m lerobot.teleoperate \
            --robot.type=bi_so101_follower \
            --robot.left_arm_port=COM4 \
            --robot.right_arm_port=COM9 \
            --robot.id=my_bimanual \
            --teleop.type=bi_so101_leader \
            --teleop.left_arm_port=COM11 \
            --teleop.right_arm_port=COM3 \
            --teleop.id=my_bimanual_leader \
            --display_data=true
        ;;
    2)
        echo "Starting bimanual teleoperation with RealSense cameras (RGB only)..."
        python -m lerobot.teleoperate \
            --robot.type=bi_so101_follower \
            --robot.left_arm_port=COM4 \
            --robot.right_arm_port=COM9 \
            --robot.id=my_bimanual \
            --robot.cameras='{
              "left_cam": {"type": "intelrealsense", "serial_number_or_name": "218622270973", "width": 848, "height": 480, "fps": 30},
              "right_cam": {"type": "intelrealsense", "serial_number_or_name": "218622278797", "width": 848, "height": 480, "fps": 30}
            }' \
            --teleop.type=bi_so101_leader \
            --teleop.left_arm_port=COM11 \
            --teleop.right_arm_port=COM3 \
            --teleop.id=my_bimanual_leader \
            --display_data=true
        ;;
    3)
        echo "Starting bimanual teleoperation with RealSense cameras (RGB + Depth)..."
        echo "Using simplified depth configuration:"
        echo "- Automatic depth alignment and colorization (JET, 200-5000mm)"
        echo "- Raw depth for Rerun 3D visualization"
        echo "- Colorized depth for dataset storage"
        python -m lerobot.teleoperate \
            --robot.type=bi_so101_follower \
            --robot.left_arm_port=COM4 \
            --robot.right_arm_port=COM9 \
            --robot.id=my_bimanual \
            --robot.cameras='{
              "left_cam": {
                "type": "intelrealsense", 
                "serial_number_or_name": "218622270973", 
                "width": 848, "height": 480, "fps": 30, 
                "use_depth": true
              },
              "right_cam": {
                "type": "intelrealsense", 
                "serial_number_or_name": "218622278797", 
                "width": 848, "height": 480, "fps": 30, 
                "use_depth": true
              }
            }' \
            --teleop.type=bi_so101_leader \
            --teleop.left_arm_port=COM11 \
            --teleop.right_arm_port=COM3 \
            --teleop.id=my_bimanual_leader \
            --display_data=true
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac