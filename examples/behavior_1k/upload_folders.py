import argparse
from pathlib import Path

from huggingface_hub import HfApi, upload_large_folder


def main():
    parser = argparse.ArgumentParser(
        description="Upload a folder to Hugging Face Hub using upload_large_folder"
    )
    parser.add_argument(
        "--folder-path",
        type=str,
        required=False,
        help="Path to the folder to upload (used if task-id is not provided)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        help="Repository ID on Hugging Face Hub (e.g., 'username/repo-name'). If task-id is provided, will be constructed as '{hf-user}/behavior1k-task{task_id:04d}'",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=False,
        help="Task index to upload (e.g., 0, 1, 2, ...). When provided, folder-path is constructed from root-path.",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        required=False,
        help="Root path containing task folders (e.g., /fsx/user/behavior1k-v3). Used with --task-id to construct folder path.",
    )
    parser.add_argument(
        "--hf-user",
        type=str,
        default=None,
        help="Hugging Face username for constructing repo-id with task-id (default: from HF_USER env var or 'fracapuano')",
    )
    parser.add_argument(
        "--create-repo", action="store_true", help="Create the repository if it doesn't exist"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers for upload (default: 2). For I/O-bound uploads, use 1-4 to avoid network contention.",
    )

    args = parser.parse_args()

    # Construct folder path and repo ID based on task-id or use provided values
    if args.task_id is not None:
        if not args.root_path:
            raise ValueError("--root-path is required when --task-id is provided")

        task_folder_name = f"behavior1k-task{args.task_id:04d}"
        folder_path = Path(args.root_path) / task_folder_name
        repo_id = f"{args.hf_user}/{task_folder_name}"

        print(f"Task mode: uploading task {args.task_id}")
    else:
        if not args.folder_path or not args.repo_id:
            raise ValueError(
                "Either --task-id with --root-path, or both --folder-path and --repo-id must be provided"
            )

        folder_path = Path(args.folder_path)
        repo_id = args.repo_id

    # Validate folder path
    if not folder_path.exists():
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    print(f"Uploading folder: {folder_path}")
    print(f"Repository: {repo_id}")

    # Create repository if requested
    if args.create_repo:
        api = HfApi()
        print(f"Creating repository {repo_id}...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            print("Repository created or already exists. Updating its contents")
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")

    # Upload the folder
    print(f"Starting upload with {args.num_workers} parallel workers...")
    try:
        result = upload_large_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="dataset",
            num_workers=args.num_workers,
        )
        print("✓ Upload completed successfully!")
        print(f"Commit URL: {result}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
