#!/bin/bash

# Dataset recording script for bi-manual SO101 robot with cameras

echo "=========================================="
echo "LeRobot Dataset Recording"
echo "=========================================="
echo ""
echo "Select recording configuration:"
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

# RealSense serial numbers
REALSENSE1_SERIAL="218622270973"
REALSENSE2_SERIAL="218622278797"

# Recording parameters
echo ""
echo "Recording Configuration:"
echo "======================="
read -p "Enter number of episodes to record (default: 5): " NUM_EPISODES
NUM_EPISODES=${NUM_EPISODES:-5}

read -p "Enter episode duration in seconds (default: 60): " EPISODE_TIME
EPISODE_TIME=${EPISODE_TIME:-60}

read -p "Enter reset time in seconds (default: 60): " RESET_TIME
RESET_TIME=${RESET_TIME:-60}

read -p "Enter task description (e.g., 'Grab the black cube'): " TASK_DESCRIPTION
TASK_DESCRIPTION=${TASK_DESCRIPTION:-"Grab the black cube"}

# Get Hugging Face username
echo ""
echo "Hugging Face Configuration:"
echo "==========================="
HF_USER=$(hf auth whoami 2>/dev/null | head -n 1)
if [ -z "$HF_USER" ]; then
    echo "Warning: Not logged into Hugging Face. Please run:"
    echo "hf auth login --token YOUR_TOKEN"
    echo ""
    read -p "Enter your Hugging Face username: " HF_USER
fi

read -p "Enter dataset name (default: so101_test): " DATASET_NAME
DATASET_NAME=${DATASET_NAME:-so101_test}

# Replace spaces with underscores and remove special characters for valid repo name
DATASET_NAME=$(echo "$DATASET_NAME" | sed 's/ /_/g' | sed 's/[^a-zA-Z0-9_-]//g')
echo "Sanitized dataset name: $DATASET_NAME"

DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

echo ""
echo "Recording Configuration Summary:"
echo "================================"
echo "Episodes: $NUM_EPISODES"
echo "Episode duration: ${EPISODE_TIME}s"
echo "Reset time: ${RESET_TIME}s"
echo "Task: $TASK_DESCRIPTION"
echo "Dataset repo: $DATASET_REPO_ID"
echo ""

read -p "Press Enter to start recording or Ctrl+C to cancel..."

case $choice in
    1)
        echo ""
        echo "Recording: Kinect RGB only"
        echo "=========================="
        python -m lerobot.record \
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
            --display_data=true \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"
        ;;

    2)
        echo ""
        echo "Recording: Kinect RGB + Depth"
        echo "============================="
        python -m lerobot.record \
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
            --display_data=true \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"
        ;;

    3)
        echo ""
        echo "Recording: 2 RealSense RGB only"
        echo "================================"
        python -m lerobot.record \
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
            --display_data=true \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"
        ;;

    4)
        echo ""
        echo "Recording: 2 RealSense RGB + Depth"
        echo "=================================="
        python -m lerobot.record \
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
            --display_data=true \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"
        ;;

    5)
        echo ""
        echo "Recording: 2 RealSense RGB + Kinect RGB"
        echo "======================================="
        python -m lerobot.record \
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
            --display_data=true \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"
        ;;

    6)
        echo ""
        echo "Recording: 2 RealSense RGB + Depth + Kinect RGB + Depth"
        echo "========================================================"
        python -m lerobot.record \
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
            --display_data=false \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION" \
            --verbose_camera_logs=true
        ;;

    7)
        echo ""
        echo "Recording: No cameras (robot arms only)"
        echo "======================================="
        python -m lerobot.record \
            --robot.type=$ROBOT_TYPE \
            --robot.id=$ROBOT_ID \
            --robot.left_arm_port=$LEFT_ARM_PORT \
            --robot.right_arm_port=$RIGHT_ARM_PORT \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --display_data=true \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"
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
echo "Recording completed!"
echo "Dataset uploaded to: https://huggingface.co/datasets/$DATASET_REPO_ID"
echo ""
echo "Next steps:"
echo "1. Train your policy: python -m lerobot.scripts.train --dataset.repo_id=$DATASET_REPO_ID --policy.type=act"
echo "2. Evaluate your policy: python -m lerobot.record --policy.path=YOUR_POLICY_PATH --dataset.repo_id=${HF_USER}/eval_${DATASET_NAME}"
