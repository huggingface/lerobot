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
# SERIAL (default 1): chain the jobs with --dependency=afterany so SLURM runs exactly ONE at a time. This
# is important for a bandwidth benchmark — concurrent jobs would share the network to the Hub/bucket and
# corrupt every throughput number. `afterany` means a failed/OOM'd job does not stall the chain. Set
# SERIAL=0 to let the scheduler run them in parallel (only for OOM-isolation testing, not for throughput).
#
# Knobs (env overrides):
#   REPO_ID, BUCKET, WARM_BUCKET, OUT_DIR, NUM_BATCHES, TIME, MEM, GPUS, SERIAL
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
SERIAL=${SERIAL:-1}             # 1 = run one job at a time (correct for bandwidth measurement)
CPU_WORKERS=${CPU_WORKERS:-8}
GPU_WORKERS=${GPU_WORKERS:-2}   # low on purpose: each cuda worker holds a CUDA context + NVDEC session
CPU_BUFFER=${CPU_BUFFER:-4000}
GPU_BUFFER=${GPU_BUFFER:-1000}  # smaller buffer bounds on-GPU frame memory
BATCH_SIZE=${BATCH_SIZE:-64}
RUN=${RUN:-python}
# CONDA_ENV=<name> runs each job via `conda run -n <name>` (no activation needed inside the dash --wrap;
# --no-capture-output streams logs live). Set this to a conda env that has a MODERN torchcodec (>=0.11)
# + datasets (>=4.7) — the default `base` env on many clusters is too old to decode AV1 / lacks CUDA.
CONDA_ENV=${CONDA_ENV:-}
if [ -n "$CONDA_ENV" ] && [ "$RUN" = "python" ]; then
  RUN="conda run --no-capture-output -n $CONDA_ENV python"
fi

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
prev_jid=""
for SOURCE in $SOURCES; do
  DATA_ROOT=$(data_root_for "$SOURCE")
  ROOTFLAG=""
  [ -n "$DATA_ROOT" ] && ROOTFLAG="--data_files_root $DATA_ROOT"
  for MODE in $MODES; do
    for DECODE in $DECODES; do
      if [ "$DECODE" = cpu ]; then W=$CPU_WORKERS; B=$CPU_BUFFER; else W=$GPU_WORKERS; B=$GPU_BUFFER; fi
      # Run strictly after the previous job so only one job touches the network at a time.
      DEPFLAG=""
      if [ "$SERIAL" = 1 ] && [ -n "$prev_jid" ]; then DEPFLAG="--dependency=afterany:$prev_jid"; fi
      jid=$(sbatch --parsable \
        --job-name="bench_${SOURCE}_${MODE}_${DECODE}" \
        --nodes=1 --ntasks=1 --gpus="$GPUS" --cpus-per-task=$((W + 4)) \
        --mem="$MEM" --time="$TIME" --output="$REPO_DIR/logs/%x-%j.out" \
        $DEPFLAG \
        ${ACCOUNT:+--account=$ACCOUNT} ${PARTITION:+--partition=$PARTITION} ${QOS:+--qos=$QOS} \
        --wrap "cd '$REPO_DIR' && \
          export TOKENIZERS_PARALLELISM=false && export HF_HOME=\${HF_HOME:-\$SCRATCH/hf_home} && \
          $RUN benchmarks/streaming/benchmark_streaming.py \
            --repo_id $REPO_ID $ROOTFLAG \
            --mode $MODE --source $SOURCE --video_decode_device $DECODE \
            --batch_size $BATCH_SIZE --num_workers $W --buffer_size $B \
            --num_batches $NUM_BATCHES --out_dir $OUT_DIR")
      jid=${jid%%;*}  # strip ';cluster' suffix on federated setups
      echo "submitted job $jid  bench_${SOURCE}_${MODE}_${DECODE}${DEPFLAG:+  (after $prev_jid)}"
      prev_jid=$jid
      n=$((n + 1))
    done
  done
done

echo
echo "Submitted $n jobs ($([ "$SERIAL" = 1 ] && echo 'serial chain — one runs at a time' || echo 'parallel'))."
echo "Watch:  squeue -u \$USER         (later jobs show reason '(Dependency)' until their turn)"
echo "Results: $OUT_DIR/<source>_<mode>_bs${BATCH_SIZE}_w<workers>_<decode>.{json,csv}"
echo "Summarize when done:  $RUN benchmarks/streaming/summarize_results.py $OUT_DIR"
