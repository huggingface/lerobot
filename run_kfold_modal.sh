#!/bin/bash
# Script to run k-fold cross validation on Modal Labs

# Configuration
DATASET_REPO_ID="jackvial/merged_datasets_test_2"
K_FOLDS=5
TRAINING_STEPS=100000
BATCH_SIZE=8
LEARNING_RATE=1e-5
GPU_TYPE="a10g"  # Options: t4, a10g, a100

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting K-Fold Cross Validation Pipeline${NC}"
echo "Dataset: $DATASET_REPO_ID"
echo "K-folds: $K_FOLDS"
echo "Training steps: $TRAINING_STEPS"
echo "GPU type: $GPU_TYPE"
echo ""

# Step 1: Generate k-fold splits
echo -e "${GREEN}Step 1: Generating k-fold splits...${NC}"
python kfold_split_generator.py \
    --dataset_repo_id=$DATASET_REPO_ID \
    --k=$K_FOLDS \
    --output_dir=kfold_splits \
    --seed=42

if [ $? -ne 0 ]; then
    echo "Failed to generate k-fold splits"
    exit 1
fi

# Step 2: Install Modal (if not already installed)
echo -e "${GREEN}Step 2: Checking Modal installation...${NC}"
if ! command -v modal &> /dev/null; then
    echo "Installing Modal..."
    pip install modal
fi

# Step 3: Authenticate with Modal (if needed)
echo -e "${GREEN}Step 3: Checking Modal authentication...${NC}"
modal token set --token-id $(modal token list | grep -v "Token ID" | head -1 | awk '{print $1}') 2>/dev/null || {
    echo "Please authenticate with Modal:"
    modal token new
}

# Step 4: Run k-fold training on Modal
echo -e "${GREEN}Step 4: Launching k-fold training on Modal...${NC}"

# Export wandb API key if available
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "WandB API key detected, will log to WandB"
    WANDB_ARG="--wandb-api-key=$WANDB_API_KEY"
else
    echo "No WandB API key found, running without WandB logging"
    WANDB_ARG=""
fi

# Run the Modal app
modal run modal_kfold_training.py \
    --dataset-repo-id=$DATASET_REPO_ID \
    --k=$K_FOLDS \
    --steps=$TRAINING_STEPS \
    --batch-size=$BATCH_SIZE \
    --learning-rate=$LEARNING_RATE \
    --gpu-type=$GPU_TYPE \
    $WANDB_ARG

if [ $? -eq 0 ]; then
    echo -e "${GREEN}K-fold training completed successfully!${NC}"
    
    # Step 5: Download results
    echo -e "${GREEN}Step 5: Downloading results...${NC}"
    mkdir -p kfold_results
    modal volume get lerobot-kfold-data outputs/ ./kfold_results/
    
    echo -e "${GREEN}Results downloaded to ./kfold_results/${NC}"
    echo ""
    echo "Summary file: ./kfold_results/kfold_summary.json"
    
    # Display summary if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "K-Fold Cross Validation Summary:"
        jq '.metrics.mean' ./kfold_results/kfold_summary.json
    fi
else
    echo "K-fold training failed"
    exit 1
fi 