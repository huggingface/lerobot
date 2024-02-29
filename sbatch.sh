#!/bin/bash
#SBATCH --nodes=1            # total number of nodes (N to be defined)
#SBATCH --ntasks-per-node=1  # number of tasks per node (here 8 tasks, or 1 task per GPU)
#SBATCH --gres=gpu:1         # number of GPUs reserved per node (here 8, or all the GPUs)
#SBATCH --cpus-per-task=8    # number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/rcadene/slurm/%j.out
#SBATCH --error=/home/rcadene/slurm/%j.err
#SBATCH --qos=medium
#SBATCH --mail-user=re.cadene@gmail.com
#SBATCH --mail-type=ALL

CMD=$@
echo "command: $CMD"

apptainer exec --nv \
~/apptainer/nvidia_cuda:12.2.2-devel-ubuntu22.04.sif $SHELL

source ~/.bashrc
conda activate fowm

srun $CMD
