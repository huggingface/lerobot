#!/bin/bash

# Quick test to verify the fix for task_indices length mismatch
# This should now work correctly even with --num-samples < full dataset length

echo "Testing annotate_pgen.py with --num-samples=100 on full dataset..."

python examples/dataset/annotate_pgen.py \
    --data-dir /fsx/jade_choghari/.cache/huggingface/lerobot/lerobot/svla_so101_pickplace \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --num-samples 100 \
    --sample-interval 1.0 \
    --output-dir /fsx/jade_choghari/outputs/pgen_test_fixed

if [ $? -eq 0 ]; then
    echo "✓ SUCCESS: Script completed without errors!"
    echo ""
    echo "Verifying output..."
    
    # Check that all frames have task_index_high_level
    python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

ds = LeRobotDataset(repo_id='local_test', root='/fsx/jade_choghari/outputs/pgen_test_fixed')
print(f'Dataset has {len(ds)} frames')
print(f'Features: {list(ds.features.keys())}')

# Check that task_index_high_level exists
assert 'task_index_high_level' in ds.features, 'task_index_high_level not in features!'

# Sample some frames
for idx in [0, 50, 99, 100, 500, 1000, 11938]:
    if idx < len(ds):
        frame = ds[idx]
        task_idx = frame['task_index_high_level'].item()
        print(f'Frame {idx}: task_index_high_level = {task_idx}')

print('✓ All checks passed!')
"
else
    echo "✗ FAILED: Script exited with error code $?"
fi

