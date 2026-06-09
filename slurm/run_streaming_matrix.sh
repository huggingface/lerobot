#!/bin/bash
# Submit the FULL streaming dataloading-benchmark matrix as isolated single-GPU SLURM jobs.
#
#   sources : hub (Hub streaming) | bucket (cold HF bucket) | warmed_bucket (prewarmed HF bucket)
#   modes   : single (1 frame, all cameras) | sarm (8-step / 8s delta window)
#   decode  : cpu (torchcodec on CPU, scales with workers) | cuda (NVDEC, offloads decode to the GPU)
#
# => 3 x 2 x 2 = 12 jobs. Each runs in its OWN job (1 node, 1 GPU) so an OOM is isolated and reported
# per-job by SLURM (check `sacct -j <id> --format=JobID,State,MaxRSS,ReqMem`). Submit from a login node
# inside the repo:  bash slurm/run_streaming_matrix.sh
#
# Knobs (env overrides):
#   REPO_ID, BUCKET, WARM_BUCKET, OUT_DIR, NUM_BATCHES, TIME, MEM, GPUS
#   CPU_WORKERS / CPU_BUFFER  (cpu-decode jobs)   GPU_WORKERS / GPU_BUFFER (cuda-decode jobs, kept low to
#   bound VRAM + NVDEC sessions).   RUN  ("python" by default; set RUN="uv run python" if using uv).
#   SOURCES / MODES / DECODES  to run a subset (e.g. SOURCES="hub bucket" DECODES="cpu").
#   ACCOUNT / PARTITION / QOS  passed through to sbatch if set.
set -euo pipefail

REPO_DIR=$(git rev-parse --show-toplevel)
REPO_ID=${REPO_ID:-pepijn223/robocasa_pretrain_human300_v4}
BUCKET=${BUCKET:-hf://buckets/pepijn223/robocasa-stream}
WARM_BUCKET=${WARM_BUCKET:-hf://buckets/pepijn223/robocasa-stream-warm}
OUT_DIR=${OUT_DIR:-benchmarks/streaming/results}
NUM_BATCHES=${NUM_BATCHES:-200}
TIME=${TIME:-01:00:00}
MEM=${MEM:-64G}
GPUS=${GPUS:-1}
CPU_WORKERS=${CPU_WORKERS:-8}
GPU_WORKERS=${GPU_WORKERS:-2}   # low on purpose: each cuda worker holds a CUDA context + NVDEC session
CPU_BUFFER=${CPU_BUFFER:-4000}
GPU_BUFFER=${GPU_BUFFER:-1000}  # smaller buffer bounds on-GPU frame memory
BATCH_SIZE=${BATCH_SIZE:-64}
RUN=${RUN:-python}

SOURCES=${SOURCES:-"hub bucket warmed_bucket"}
MODES=${MODES:-"single sarm"}
DECODES=${DECODES:-"cpu cuda"}

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/$OUT_DIR"

data_root_for () {
  case "$1" in
    hub) echo "" ;;
    bucket) echo "$BUCKET" ;;
    warmed_bucket) echo "$WARM_BUCKET" ;;
  esac
}

n=0
for SOURCE in $SOURCES; do
  DATA_ROOT=$(data_root_for "$SOURCE")
  ROOTFLAG=""
  [ -n "$DATA_ROOT" ] && ROOTFLAG="--data_files_root $DATA_ROOT"
  for MODE in $MODES; do
    for DECODE in $DECODES; do
      if [ "$DECODE" = cpu ]; then W=$CPU_WORKERS; B=$CPU_BUFFER; else W=$GPU_WORKERS; B=$GPU_BUFFER; fi
      sbatch \
        --job-name="bench_${SOURCE}_${MODE}_${DECODE}" \
        --nodes=1 --ntasks=1 --gpus="$GPUS" --cpus-per-task=$((W + 4)) \
        --mem="$MEM" --time="$TIME" --output="$REPO_DIR/logs/%x-%j.out" \
        ${ACCOUNT:+--account=$ACCOUNT} ${PARTITION:+--partition=$PARTITION} ${QOS:+--qos=$QOS} \
        --wrap "set -euo pipefail; cd '$REPO_DIR'; \
          export TOKENIZERS_PARALLELISM=false HF_HOME=\${HF_HOME:-\$SCRATCH/hf_home}; \
          $RUN benchmarks/streaming/benchmark_streaming.py \
            --repo_id $REPO_ID $ROOTFLAG \
            --mode $MODE --source $SOURCE --video_decode_device $DECODE \
            --batch_size $BATCH_SIZE --num_workers $W --buffer_size $B \
            --num_batches $NUM_BATCHES --out_dir $OUT_DIR"
      n=$((n + 1))
    done
  done
done

echo "Submitted $n jobs. Watch:  squeue -u \$USER"
echo "Results land in $OUT_DIR/<source>_<mode>_bs${BATCH_SIZE}_w<workers>_<decode>.{json,csv}"
echo "After they finish, summarize:  python benchmarks/streaming/summarize_results.py $OUT_DIR"
