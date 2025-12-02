python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id lerobot \
        --root /fsx/jade_choghari/vlabench-primitive \
        --operation.type convert_to_video \
        --operation.output_dir /fsx/jade_choghari/vlabench-primitive-encoded \
        --operation.num_workers 10 \
        --operation.vcodec libsvtav1
