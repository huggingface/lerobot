#!/bin/bash

# Simple teleoperation script for bi-manual SO101 robot with cameras

echo "=========================================="
echo "LeRobot Camera Teleoperation"
echo "=========================================="
echo ""
echo "Select test configuration:"
echo "1) Kinect RGB only"
echo "2) Kinect RGB + Depth"
echo "3) 2 RealSense RGB only"
echo "4) 2 RealSense RGB + Depth"
echo "5) 2 RealSense RGB + Kinect RGB"
echo "6) 2 RealSense RGB + Depth + Kinect RGB + Depth"
echo "7) No cameras (robot arms only)"
echo "8) Exit"
echo ""
read -p "Enter your choice (1-8): " choice

# Activate conda environment
echo "Activating lerobot environment..."
eval "$(conda shell.bash hook)"
conda activate lerobot

# Common parameters
ROBOT_TYPE="bi_so101_follower"
ROBOT_ID="my_bimanual"
LEFT_ARM_PORT="COM4"
RIGHT_ARM_PORT="COM9"
TELEOP_TYPE="bi_so101_leader"
TELEOP_ID="my_bimanual_leader"
TELEOP_LEFT_PORT="COM11"
TELEOP_RIGHT_PORT="COM3"
FPS="30"

# RealSense serial numbers
REALSENSE1_SERIAL="218622270973"
REALSENSE2_SERIAL="218622278797"

case $choice in
    1)
        echo ""
        echo "Running: Kinect RGB only"
        echo "========================"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_main': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'use_depth': false,
                    'pipeline': 'cuda',
                    'enable_bilateral_filter': false,
                    'enable_edge_filter': false
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=true
        ;;

    2)
        echo ""
        echo "Running: Kinect RGB + Depth"
        echo "==========================="
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_main': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.5,
                    'depth_max_meters': 3.0,
                    'depth_clipping': true,
                    'enable_bilateral_filter': false,
                    'enable_edge_filter': false,
                    'pipeline': 'cuda'
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=true
        ;;

    3)
        echo ""
        echo "Running: 2 RealSense RGB only"
        echo "=============================="
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_low': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE1_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30
                },
                'cam_high': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE2_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=true
        ;;

    4)
        echo ""
        echo "Running: 2 RealSense RGB + Depth"
        echo "================================="
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_low': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE1_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.3,
                    'depth_max_meters': 3.0,
                    'depth_clipping': true
                },
                'cam_high': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE2_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.3,
                    'depth_max_meters': 3.0,
                    'depth_clipping': true
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=true
        ;;

    5)
        echo ""
        echo "Running: 2 RealSense RGB + Kinect RGB"
        echo "======================================"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_low': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE1_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30
                },
                'cam_high': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE2_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30
                },
                'cam_kinect': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'use_depth': false,
                    'pipeline': 'cuda',
                    'enable_bilateral_filter': false,
                    'enable_edge_filter': false
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=true
        ;;

    6)
        echo ""
        echo "Running: 2 RealSense RGB + Depth + Kinect RGB + Depth"
        echo "======================================================"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_low': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE1_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.3,
                    'depth_max_meters': 3.0,
                    'depth_clipping': true
                },
                'cam_high': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE2_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.3,
                    'depth_max_meters': 3.0,
                    'depth_clipping': true
                },
                'cam_kinect': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.5,
                    'depth_max_meters': 3.0,
                    'depth_clipping': true,
                    'enable_bilateral_filter': false,
                    'enable_edge_filter': false,
                    'pipeline': 'cuda'
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=false
        ;;

    7)
        echo ""
        echo "Running: No cameras (robot arms only)"
        echo "======================================"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=true
        ;;

    8)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "Invalid choice. Please select 1-8."
        exit 1
        ;;
esac

echo ""
echo "Test completed!"
