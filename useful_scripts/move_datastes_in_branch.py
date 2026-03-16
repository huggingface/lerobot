import os
import shutil
import tempfile

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

# --- CONFIGURATION ---
# Set this in your shell before running:
# export HF_TOKEN="hf_xxx"
TOKEN = ""
MASTER_REPO_ID = "HSP-IIT/stacking"
DELETE_SOURCE_REPOS = True
DELETE_CONFIRMATION = "DELETE"
REPOS_TO_MIGRATE = [
    "HSP-IIT/fruit"
]

# Files that may exist in target dataset branches even if absent in source.
ALLOWED_TARGET_ONLY_FILES = {"README.md", ".gitattributes"}


def ensure_config() -> None:
    if not TOKEN:
        raise ValueError("HF_TOKEN environment variable is required.")
    if not MASTER_REPO_ID or "/" not in MASTER_REPO_ID:
        raise ValueError("MASTER_REPO_ID must look like 'org_or_user/repo_name'.")
    if DELETE_SOURCE_REPOS and DELETE_CONFIRMATION != "DELETE":
        raise ValueError(
            "Deletion is enabled but not confirmed. Set DELETE_CONFIRMATION = 'DELETE'."
        )


def _repo_file_set(api: HfApi, repo_id: str, revision: str | None = None) -> set[str]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    return set(files)


def verify_migration(api: HfApi, source_repo_id: str, target_repo_id: str, target_branch: str) -> bool:
    """Return True when every source file exists in target branch.

    Target branches may contain extra metadata files (e.g., README.md).
    """
    source_files = _repo_file_set(api, source_repo_id, revision=None)
    target_files = _repo_file_set(api, target_repo_id, revision=target_branch)

    missing_in_target = sorted(source_files - target_files)
    extra_in_target = sorted(target_files - source_files)
    non_allowed_extras = [f for f in extra_in_target if f not in ALLOWED_TARGET_ONLY_FILES]

    if missing_in_target or non_allowed_extras:
        print("  [VERIFY][ERROR] Source/target file mismatch")
        if missing_in_target:
            print(f"    Missing in target (first 20): {missing_in_target[:20]}")
        if non_allowed_extras:
            print(f"    Unexpected extra in target (first 20): {non_allowed_extras[:20]}")
        return False

    allowed_extras = [f for f in extra_in_target if f in ALLOWED_TARGET_ONLY_FILES]
    if allowed_extras:
        print(f"  [VERIFY][INFO] Allowed target-only files: {allowed_extras[:20]}")
    print(f"  [VERIFY][OK] All source files present in target ({len(source_files)} files)")
    return True


def migrate_repos() -> None:
    ensure_config()
    api = HfApi(token=TOKEN)

    for repo_id in REPOS_TO_MIGRATE:
        branch_name = repo_id.split("/")[-1]
        tmp_dir = None
        print(f"\n[START] Migrating {repo_id} -> {MASTER_REPO_ID}@{branch_name}")

        try:
            tmp_dir = tempfile.mkdtemp(prefix=f"hf_migrate_{branch_name}_")

            print(f"  [1/3] Downloading source dataset: {repo_id}")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False,
                token=TOKEN,
                ignore_patterns=[".git*", ".gitattributes"],
            )

            print(f"  [2/3] Ensuring destination branch exists: {branch_name}")
            try:
                api.create_branch(
                    repo_id=MASTER_REPO_ID,
                    repo_type="dataset",
                    branch=branch_name,
                )
                print(f"        Created branch '{branch_name}'")
            except HfHubHTTPError as err:
                status = getattr(err.response, "status_code", None)
                if status == 409:
                    print(f"        Branch '{branch_name}' already exists")
                else:
                    raise

            print(f"  [3/4] Uploading snapshot to {MASTER_REPO_ID}@{branch_name}")
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=MASTER_REPO_ID,
                repo_type="dataset",
                revision=branch_name,
                commit_message=f"Migrate from {repo_id}",
            )

            print(f"  [OK] Migration completed for {repo_id}")

            print("  [4/4] Verifying migrated files before any deletion")
            migration_verified = verify_migration(
                api=api,
                source_repo_id=repo_id,
                target_repo_id=MASTER_REPO_ID,
                target_branch=branch_name,
            )
            if not migration_verified:
                raise RuntimeError("Verification failed, source repo will not be deleted.")

            if DELETE_SOURCE_REPOS:
                if repo_id == MASTER_REPO_ID:
                    raise RuntimeError("Refusing to delete source because it matches MASTER_REPO_ID")
                print(f"  [DELETE] Deleting source dataset repo: {repo_id}")
                api.delete_repo(repo_id=repo_id, repo_type="dataset")
                print(f"  [OK] Deleted source dataset repo: {repo_id}")

        except Exception as err:
            print(f"  [ERROR] Migration failed for {repo_id}: {err}")
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                print("  [CLEANUP] Temporary files removed")


if __name__ == "__main__":
    migrate_repos()