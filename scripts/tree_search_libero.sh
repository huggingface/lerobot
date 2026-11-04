uv run python examples/tree_search/policy_inference_api.py \
    --policy.path=lerobot/pi0_libero_finetuned_v044 \
    --env.type=libero \
    --env.task=libero_10 \
    --task_id=0 \
    --steps=0 \
    --policy.device=mps
