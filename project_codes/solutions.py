"""
WORKAROUND SOLUTIONS FOR MERGED DATASET ISSUES

Problem: The merged dataset has 68 bad episodes out of 180 due to video frame index mismatches.

Solutions ranked by difficulty:
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def solution_1_use_good_episodes_only():
    """
    Solution 1: Use only the good episodes (EASIEST)

    You have 112 good episodes. Use them for training by specifying the episode list.
    """
    print("="*60)
    print("SOLUTION 1: Use Only Good Episodes")
    print("="*60)
    print("\nWhen training, use this configuration:\n")

    good_episodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65,
                     66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179]

    print(f"dataset.episodes={good_episodes}")
    print(f"\nOR in your training command:")
    print(f'--dataset.episodes="[{",".join(map(str, good_episodes))}]"')
    print(
        f"\nThis gives you {len(good_episodes)} out of 180 episodes (62% of data)")


def solution_2_use_separate_datasets():
    """
    Solution 2: Don't merge - use both datasets separately (RECOMMENDED)

    Both original datasets work fine. Train on them separately or configure
    your training to use multiple datasets.
    """
    print("\n" + "="*60)
    print("SOLUTION 2: Use Separate Datasets (RECOMMENDED)")
    print("="*60)
    print("\nInstead of merging, use both datasets independently:\n")
    print("Right dataset: YieumYoon/bimanual-center-basket-right-rblock")
    print("  - 90 episodes, 51,426 frames")
    print("  - ALL episodes work!")
    print("\nLeft dataset: YieumYoon/bimanual-center-basket-left-rblock")
    print("  - 90 episodes, 53,736 frames")
    print("  - ALL episodes work!")
    print("\nBenefits:")
    print("  ✓ 100% of your data (180 episodes)")
    print("  ✓ No corrupted episodes")
    print("  ✓ Can balance sampling between datasets")
    print("\nExample training config:")
    print("""
    datasets:
      - repo_id: YieumYoon/bimanual-center-basket-right-rblock
        weight: 0.5
      - repo_id: YieumYoon/bimanual-center-basket-left-rblock
        weight: 0.5
    """)


def solution_3_try_pyav_backend():
    """
    Solution 3: Try PyAV video backend (EXPERIMENTAL)

    Different video backends handle timestamps differently. PyAV might work better.
    """
    print("\n" + "="*60)
    print("SOLUTION 3: Try PyAV Backend (EXPERIMENTAL)")
    print("="*60)
    print("\nTest if PyAV backend handles the timestamps better:\n")
    print("ds = LeRobotDataset(")
    print('    repo_id="YieumYoon/bimanual-center-basket-rblock-rlmerged",')
    print('    video_backend="pyav"  # Instead of "torchcodec"')
    print(")")
    print("\nNote: This may or may not fix the issue, but worth trying.")


def solution_4_delete_and_remerge():
    """
    Solution 4: Delete cache and re-merge (NUCLEAR OPTION)

    Sometimes the cache gets corrupted. Delete and try again.
    """
    print("\n" + "="*60)
    print("SOLUTION 4: Clear Cache and Re-merge")
    print("="*60)
    print("\nSteps:")
    print("1. Delete the merged dataset cache:")
    print("   rm -rf ~/.cache/huggingface/lerobot/YieumYoon/bimanual-center-basket-rblock-rlmerged*")
    print("\n2. Re-run the merge command:")
    print("""
   lerobot-edit-dataset \\
       --repo_id YieumYoon/bimanual-center-basket-rblock-rlmerged-3 \\
       --operation.type merge \\
       --operation.repo_ids "['YieumYoon/bimanual-center-basket-right-rblock', \\
                              'YieumYoon/bimanual-center-basket-left-rblock']" \\
       --push_to_hub=false  # Test locally first
    """)
    print("\n3. Validate the new merged dataset before pushing:")
    print("   python eval_dataset.py  # Should show 0 bad episodes")


def solution_5_file_bug_report():
    """
    Solution 5: Report the bug to LeRobot team

    This appears to be a bug in the aggregate_datasets function.
    """
    print("\n" + "="*60)
    print("SOLUTION 5: File a Bug Report")
    print("="*60)
    print("\nThis appears to be a bug in LeRobot's dataset merging.")
    print("\nWhat to report:")
    print("  - aggregate_datasets() produces corrupted frame indices")
    print("  - Video concatenation works, but metadata doesn't match")
    print("  - Frame index errors like: 'Invalid frame index=51426; must be less than 33231'")
    print("\nWhere to report:")
    print("  https://github.com/huggingface/lerobot/issues")
    print("\nInclude:")
    print("  - Your merge command")
    print("  - The error messages")
    print("  - Dataset repo IDs")


def main():
    print("\n" + "="*70)
    print(" SOLUTIONS FOR MERGED DATASET PROBLEMS")
    print("="*70)

    solution_1_use_good_episodes_only()
    solution_2_use_separate_datasets()
    solution_3_try_pyav_backend()
    solution_4_delete_and_remerge()
    solution_5_file_bug_report()

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\n🎯 Best solution: Use SOLUTION 2 (separate datasets)")
    print("   - Keeps 100% of your data")
    print("   - No corruption issues")
    print("   - Clean and reliable")
    print("\n📊 Quick fix: Use SOLUTION 1 (good episodes only)")
    print("   - Loses 38% of data but works immediately")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
