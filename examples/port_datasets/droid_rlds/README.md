# Port DROID 1.0.1 dataset to LeRobotDataset

## Download

TODO

It will take 2 TB in your local disk.

## Port on a single computer

First, install tensorflow dataset utilities to read from raw files:
```bash
pip install tensorflow
pip install tensorflow_datasets
```

Then run this script to start porting the dataset:
```bash
python examples/port_datasets/droid_rlds/port_droid.py \
    --raw-dir /your/data/droid/1.0.1 \
    --repo-id your_id/droid_1.0.1 \
    --push-to-hub
```

It will take 400GB in your local disk.

As usual, your LeRobotDataset will be stored in your huggingface/lerobot cache folder.

WARNING: it will take 7 days for porting the dataset locally and 3 days to upload, so we will need to parallelize over multiple nodes on a slurm cluster.

NOTE: For development, run this script to start porting a shard:
```bash
python examples/port_datasets/droid_rlds/port.py \
    --raw-dir /your/data/droid/1.0.1 \
    --repo-id your_id/droid_1.0.1 \
    --num-shards 2048 \
    --shard-index 0
```

## Port over SLURM

Install slurm utilities from Hugging Face:
```bash
pip install datatrove
```


### 1. Port one shard per job

Run this script to start porting shards of the dataset:
```bash
python examples/port_datasets/droid_rlds/slurm_port_shards.py \
    --raw-dir /your/data/droid/1.0.1 \
    --repo-id your_id/droid_1.0.1 \
    --logs-dir /your/logs \
    --job-name port_droid \
    --partition your_partition \
    --workers 2048 \
    --cpus-per-task 8 \
    --mem-per-cpu 1950M
```

**Note on how to set your command line arguments**

Regarding `--partition`, find yours by running:
```bash
info --format="%R"`
```
and select the CPU partition if you have one. No GPU needed.

Regarding `--workers`, it is the number of slurm jobs you will launch in parallel. 2048 is the maximum number, since there is 2048 shards in Droid. This big number will certainly max-out your cluster.

Regarding `--cpus-per-task` and `--mem-per-cpu`, by default it will use ~16GB of RAM (8*1950M) which is recommended to load the raw frames and 8 CPUs which can be useful to parallelize the encoding of the frames.

Find the number of CPUs and Memory of the nodes of your partition by running:
```bash
sinfo -N -p your_partition -h -o "%N cpus=%c mem=%m"
```

**Useful commands to check progress and debug**

Check if your jobs are running:
```bash
squeue -u $USER`
```

You should see a list with job indices like `15125385_155` where `15125385` is the index of the run and `155` is the worker index. The output/print of this worker is written in real time in `/your/logs/job_name/slurm_jobs/15125385_155.out`. For instance, you can inspect the content of this file by running `less /your/logs/job_name/slurm_jobs/15125385_155.out`.

Check the progression of your jobs by running:
```bash
jobs_status /your/logs
```

If it's not 100% and no more slurm job is running, it means that some of them failed. Inspect the logs by running:
```bash
failed_logs /your/logs/job_name
```

If there is an issue in the code, you can fix it in debug mode with `--slurm 0` which allows to set breakpoint:
```bash
python examples/port_datasets/droid_rlds/slurm_port_shards.py --slurm 0 ...
```

And you can relaunch the same command, which will skip the completed jobs:
```bash
python examples/port_datasets/droid_rlds/slurm_port_shards.py --slurm 1 ...
```

Once all jobs are completed, you will have one dataset per shard (e.g. `droid_1.0.1_world_2048_rank_1594`) saved on disk in your `/lerobot/home/dir/your_id` directory. You can find your `/lerobot/home/dir` by running:
```bash
python -c "from lerobot.common.constants import HF_LEROBOT_HOME;print(HF_LEROBOT_HOME)"
```


### 2. Aggregate all shards

Run this script to start aggregation:
```bash
python examples/port_datasets/droid_rlds/slurm_aggregate_shards.py \
    --repo-id your_id/droid_1.0.1 \
    --logs-dir /your/logs \
    --job-name aggr_droid \
    --partition your_partition \
    --workers 2048 \
    --cpus-per-task 8 \
    --mem-per-cpu 1950M
```

Once all jobs are completed, you will have one dataset your `/lerobot/home/dir/your_id/droid_1.0.1` directory.


### 3. Upload dataset

Run this script to start uploading:
```bash
python examples/port_datasets/droid_rlds/slurm_upload.py \
    --repo-id your_id/droid_1.0.1 \
    --logs-dir /your/logs \
    --job-name upload_droid \
    --partition your_partition \
    --workers 50 \
    --cpus-per-task 4 \
    --mem-per-cpu 1950M
```
