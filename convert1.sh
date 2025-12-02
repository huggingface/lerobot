python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot \
        --root /fsx/jade_choghari/vlabench-composite \
        --operation.type convert_to_video \
        --operation.output_dir /fsx/jade_choghari/vlabench-composite-encoded \
        --operation.num_workers 10 \
        --operation.vcodec libsvtav1 \
        --push_to_hub false
