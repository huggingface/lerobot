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
echo "8) 2 RealSense RGB + Depth + Kinect RGB only"
echo "9) Exit"
echo ""
read -p "Enter your choice (1-9): " choice

# Activate conda environment
echo "Activating lerobot environment..."
eval "$(conda shell.bash hook)"
conda activate lerobot

# Common parameters
ROBOT_TYPE="bi_so101_follower"
ROBOT_ID="my_bimanual"
LEFT_ARM_PORT="COM6"
RIGHT_ARM_PORT="COM3"
TELEOP_TYPE="bi_so101_leader"
TELEOP_ID="my_bimanual_leader"
TELEOP_LEFT_PORT="COM5"
TELEOP_RIGHT_PORT="COM4"
FPS=30

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

# Viz and logging options
echo ""
read -p "Enable depth visualization in viewer (viz_depth)? [y/N]: " VIZ_INPUT
if [[ "$VIZ_INPUT" =~ ^[Yy]$ ]]; then
    VIZ_DEPTH=true
else
    VIZ_DEPTH=false
fi

# Logging options
echo ""
read -p "Enable performance logging (every 5s)? [y/N]: " PERF_INPUT
if [[ "$PERF_INPUT" =~ ^[Yy]$ ]]; then
    PERF_ARGS="--perf_logging=true --perf_level=INFO"
else
    PERF_ARGS="--perf_logging=false"
fi

read -p "Enter log file path (default dir: C:/Users/madha/Documents/lerobot): " LOG_FILE
LOG_DIR="C:/Users/madha/Documents/lerobot"
if [ -z "$LOG_FILE" ]; then
    LOG_FILE="$LOG_DIR/record_$(date +%Y%m%d_%H%M%S).log"
fi
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null

# Note: No stdout/stderr redirection; perf-only logs go to file via app logging

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
echo "Viz depth: $VIZ_DEPTH"
echo "Log file: ${LOG_FILE:-none}"
echo "Perf logging: ${PERF_ARGS}"
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
                    'pipeline': 'cuda'
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --display_data=true \
            --viz_depth=$VIZ_DEPTH \
            --log_file="$LOG_FILE" \
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS
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
                    'pipeline': 'cuda'
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --display_data=true \
            --viz_depth=$VIZ_DEPTH \
            --log_file="$LOG_FILE" \
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS
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
            --viz_depth=$VIZ_DEPTH \
            --log_file="$LOG_FILE" \
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS
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
                    'depth_min_meters': 0.07,
                    'depth_max_meters': 0.5
                },
                'cam_high': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE2_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.07,
                    'depth_max_meters': 0.5
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --display_data=true \
            --viz_depth=$VIZ_DEPTH \
            --log_file="$LOG_FILE" \
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS
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
                    'pipeline': 'cuda'
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --display_data=true \
            --viz_depth=$VIZ_DEPTH \
            --log_file="$LOG_FILE" \
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS
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
                    'depth_min_meters': 0.07,
                    'depth_max_meters': 0.5,
                    
                },
                'cam_high': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE2_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.07,
                    'depth_max_meters': 0.4,
                    
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
                    'depth_max_meters': 0.8,

                    'pipeline': 'cuda'
                }
            }" \
            --teleop.type=$TELEOP_TYPE \
            --teleop.id=$TELEOP_ID \
            --teleop.left_arm_port=$TELEOP_LEFT_PORT \
            --teleop.right_arm_port=$TELEOP_RIGHT_PORT \
            --display_data=false \
            --viz_depth=$VIZ_DEPTH \
            --log_file="$LOG_FILE" \
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS $REDIRECT_STD
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
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS
        ;;

    8)
        echo ""
        echo "Recording: 2 RealSense RGB + Depth + Kinect RGB only"
        echo "===================================================="
        run_cmd python -m lerobot.record \
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
                    'depth_min_meters': 0.07,
                    'depth_max_meters': 0.5
                },
                'cam_high': {
                    'type': 'intelrealsense',
                    'serial_number_or_name': '$REALSENSE2_SERIAL',
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'use_depth': true,
                    'depth_colormap': 'jet',
                    'depth_min_meters': 0.07,
                    'depth_max_meters': 0.5
                },
                'cam_kinect': {
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
            --display_data=true \
            $PERF_ARGS \
            --dataset.repo_id=$DATASET_REPO_ID \
            --dataset.num_episodes=$NUM_EPISODES \
            --dataset.episode_time_s=$EPISODE_TIME \
            --dataset.reset_time_s=$RESET_TIME \
            --dataset.single_task="$TASK_DESCRIPTION"\
            --dataset.fps=$FPS
        ;;

    9)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "Invalid choice. Please select 1-9."
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
