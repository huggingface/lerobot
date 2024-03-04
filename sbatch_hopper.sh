#!/bin/bash
#SBATCH --nodes=1            # total number of nodes (N to be defined)
#SBATCH --ntasks-per-node=1  # number of tasks per node (here 8 tasks, or 1 task per GPU)
#SBATCH --qos=normal         # number of GPUs reserved per node (here 8, or all the GPUs)
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:1         # number of GPUs reserved per node (here 8, or all the GPUs)
#SBATCH --cpus-per-task=12    # number of cores per task
#SBATCH --mem-per-cpu=11G
#SBATCH --time=12:00:00
#SBATCH --output=/admin/home/remi_cadene/slurm/%j.out
#SBATCH --error=/admin/home/remi_cadene/slurm/%j.err
#SBATCH --mail-user=remi_cadene@huggingface.co
#SBATCH --mail-type=ALL

CMD=$@
echo "command: $CMD"
srun $CMD
