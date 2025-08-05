#!/bin/bash

# Teleoperation script for bi-manual SO101 robot with RGB camera testing

echo "=========================================="
echo "LeRobot Camera Teleoperation"
echo "=========================================="
echo ""
echo "Select camera configuration:"
echo "1) Kinect RGB only"
echo "2) Kinect RGB + Depth (side by side)"
echo "3) 2 RealSense cameras"
echo "4) RealSense + Kinect RGB"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

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

    2)
        echo ""
        echo "Running: Kinect RGB + Depth (side by side)"
        echo "==========================================="
        echo "Note: Shows both RGB and colorized depth"
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
        echo "Running: 2 RealSense cameras"
        echo "============================="
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
        echo "Running: RealSense + Kinect RGB"
        echo "================================"
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
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
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

    5)
        echo ""
        echo "Running: Kinect with colorized depth"
        echo "===================================="
        echo "Note: Shows colorized depth visualization"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_depth_jet': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 512,
                    'height': 424,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.5,
                    'depth_max_meters': 3.0,
                    'depth_clipping': true,
                    'enable_bilateral_filter': true,
                    'enable_edge_filter': true
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
        echo "Running: Mixed cameras with depth (RealSense RGB + Kinect Depth)"
        echo "================================================================"
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
                'cam_depth': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 512,
                    'height': 424,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'hot',
                    'depth_min_meters': 0.5,
                    'depth_max_meters': 2.5
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --fps=$FPS \
            --display_data=true
        ;;

    7)
        echo ""
        echo "Running: Kinect CUDA pipeline (RGB only)"
        echo "========================================"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_kinect': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
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

    8)
        echo ""
        echo "Running: Kinect with custom colormap (Viridis depth)"
        echo "===================================================="
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_depth_viridis': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 512,
                    'height': 424,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'viridis',
                    'depth_min_meters': 0.3,
                    'depth_max_meters': 4.0,
                    'depth_clipping': true,
                    'enable_bilateral_filter': true,
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

    9)
        echo ""
        echo "Running: Kinect RGB + Depth side-by-side"
        echo "========================================="
        echo "Note: Shows RGB and colorized depth from same Kinect"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_rgb': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'use_depth': false
                },
                'cam_depth_hot': {
                    'type': 'kinect',
                    'device_index': 1,
                    'width': 512,
                    'height': 424,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'hot',
                    'depth_min_meters': 0.5,
                    'depth_max_meters': 2.5,
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

    10)
        echo ""
        echo "Running: Multi-Kinect setup (if available)"
        echo "=========================================="
        echo "Note: This requires multiple Kinect v2 devices on separate USB 3.0 controllers"
        python -m lerobot.teleoperate \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --robot.cameras "{
                'cam_kinect_0': {
                    'type': 'kinect',
                    'device_index': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30
                },
                'cam_kinect_1': {
                    'type': 'kinect',
                    'device_index': 1,
                    'width': 1920,
                    'height': 1080,
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

    11)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "Invalid choice. Please select 1-11."
        exit 1
        ;;
esac

echo ""
echo "Test completed!"
