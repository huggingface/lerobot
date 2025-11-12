from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot_download(
        repo_id="behavior-1k/2025-challenge-demos",
        repo_type="dataset",
        local_dir="/fsx/francesco_capuano/behavior1k-2025-v21",
    )
