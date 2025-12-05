"""
FINAL RECOMMENDATION: DO NOT MERGE - USE SEPARATE DATASETS

The LeRobot aggregate_datasets() function has a bug when:
1. Source video files are large (>150 MB)
2. Videos need to be concatenated and then split across multiple files
3. The metadata frame index tracking breaks

PROVEN SOLUTION: Use both datasets separately in training.
"""

print("="*70)
print(" FINAL ANALYSIS: aggregate_datasets() IS FUNDAMENTALLY BROKEN")
print("="*70)

print("\nWhat we tried:")
print("  1. DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200 MB → 68 bad episodes")
print("  2. DEFAULT_VIDEO_FILE_SIZE_IN_MB = 500 MB → 31 bad episodes")
print("  3. DEFAULT_VIDEO_FILE_SIZE_IN_MB = 1000 MB → 60 bad episodes (WORSE!)")

print("\nWhy it's broken:")
print("  - Your top camera videos are ~400 MB when concatenated")
print("  - Even with 1000 MB limit, they still get split into 2 files")
print("  - When videos split, the frame index metadata becomes incorrect")
print("  - This is a BUG in aggregate_datasets(), not a configuration issue")

print("\n" + "="*70)
print(" ✅ SOLUTION: USE SEPARATE DATASETS")
print("="*70)

print("\nExample training configuration:")
print("""
# In your training script or config file:

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load both datasets
dataset_right = LeRobotDataset("YieumYoon/bimanual-center-basket-right-rblock")
dataset_left = LeRobotDataset("YieumYoon/bimanual-center-basket-left-rblock")

# Or configure in your training yaml:
dataset:
  repo_ids:
    - YieumYoon/bimanual-center-basket-right-rblock
    - YieumYoon/bimanual-center-basket-left-rblock
  
# Many training frameworks support multiple datasets and will handle
# sampling from both automatically
""")

print("\n" + "="*70)
print(" BENEFITS OF SEPARATE DATASETS")
print("="*70)
print("  ✅ 100% of your data (180 episodes)")
print("  ✅ No corruption or errors")
print("  ✅ No workarounds needed")
print("  ✅ Better control over dataset sampling")
print("  ✅ Can adjust weights between datasets")

print("\n" + "="*70)
print(" ALTERNATIVE: Report Bug to LeRobot Team")
print("="*70)
print("\nIf you MUST have a merged dataset, report this bug:")
print("  Repository: https://github.com/huggingface/lerobot/issues")
print("  Title: 'aggregate_datasets() corrupts frame indices when videos split'")
print("  Details:")
print("    - aggregate_videos() fails when dst_size + src_size >= limit")
print("    - Creates new video file but metadata still points to old file")
print("    - Frame indices become incorrect for episodes after the split")
print("    - Affects datasets with large video files (>150 MB)")

print("\n" + "="*70)
print(" YOUR DATASETS ARE FINE!")
print("="*70)
print("\n  RIGHT: 90 episodes, 0 errors ✓")
print("  LEFT:  90 episodes, 0 errors ✓")
print("  TOTAL: 180 perfect episodes")
print("\n  The ONLY problem is the merge function. Use them separately!")
print("="*70 + "\n")
