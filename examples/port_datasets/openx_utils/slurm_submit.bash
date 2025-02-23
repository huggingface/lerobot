#!/bin/bash
#SBATCH --job-name=openx_rlds
#SBATCH --partition=hopper-cpu
#SBATCH --requeue
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=/fsx/%u/slurm/%j-%x.out

OUTPUT_DIR="/fsx/${USER}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-tasks"
mkdir -p $OUTPUT_DIR

# Function to run a task and redirect output to a separate file
run_task() {
    local task_id=$1
    local output_file="${OUTPUT_DIR}/task-${task_id}-${SLURM_JOB_ID}.out"

    # Run the task and redirect output
    python examples/port_datasets/openx_utils/test.py > $output_file 2>&1
}

echo $SBATCH_OUTPUT

# node has 380850M and 96 cpus
trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' SIGUSR1
echo "Starting job"
# note the "&" to start srun as a background thread
srun python examples/port_datasets/openx_utils/test.py &
# wait for signals...
wait
