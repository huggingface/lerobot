#!/usr/bin/env bash
set -u
export ALL_PROXY=socks5h://127.0.0.1:1080 HTTPS_PROXY=socks5h://127.0.0.1:1080 HTTP_PROXY=socks5h://127.0.0.1:1080
export HF_HUB_DOWNLOAD_TIMEOUT=300 MUJOCO_GL=egl
ROOT=outputs/eval/libero_pertask
mkdir -p "$ROOT"
run_one () {
  local suite=$1 tid=$2 outdir=$3
  uv run lerobot-eval \
    --policy.path=lerobot/smolvla_libero \
    --env.type=libero \
    --env.task="$suite" \
    --env.task_ids="[$tid]" \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --eval.use_async_envs=false \
    --policy.device=cuda \
    '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
    --policy.empty_cameras=1 \
    --env.max_parallel_tasks=1 \
    --output_dir="$outdir" > "$outdir.log" 2>&1
}
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  for TID in 0 1 2 3 4 5 6 7 8 9; do
    OUT="$ROOT/${SUITE}_task${TID}"
    rm -rf "$OUT" "$OUT.log"
    echo ">>> RUN $SUITE task=$TID $(date +%H:%M:%S)"
    run_one "$SUITE" "$TID" "$OUT"
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$OUT/eval_info.json" ]; then
      echo "!!! RETRY $SUITE task=$TID (rc=$rc) $(date +%H:%M:%S)"
      rm -rf "$OUT" "$OUT.log"
      run_one "$SUITE" "$TID" "$OUT"
      rc=$?
    fi
    if [ -f "$OUT/eval_info.json" ]; then
      echo "<<< OK   $SUITE task=$TID rc=$rc"
    else
      echo "XXX FAIL $SUITE task=$TID rc=$rc (no eval_info.json)"
    fi
  done
done
echo "ALL_PERTASK_FINISHED $(date +%H:%M:%S)"
