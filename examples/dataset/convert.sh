python ./examples/dataset/convert_hdf5_lerobot.py \
    --src-paths /fsx/jade_choghari/XVLA-Soft-Fold/0808_12am_stage_1_stage2new_new_cam_very_slow_no_sleeve \
    --output-path /fsx/jade_choghari/new-data \
    --executor local \
    --tasks-per-job 3 \
    --workers 10
