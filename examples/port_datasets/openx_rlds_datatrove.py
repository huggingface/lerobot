import datetime as dt
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class PortOpenXDataset(PipelineStep):
    def __init__(
        self,
        raw_dir: Path,
        repo_id: str = None,
        image_writer_process: int = 0,
        image_writer_threads: int = 8,
    ):
        super().__init__()
        self.raw_dir = raw_dir
        self.repo_id = repo_id
        self.image_writer_process = image_writer_process
        self.image_writer_threads = image_writer_threads

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from examples.port_datasets.openx_rlds import create_lerobot_dataset
        from examples.port_datasets.openx_utils.test import display_slurm_info, display_system_info

        display_system_info()
        display_slurm_info()

        create_lerobot_dataset(
            self.raw_dir,
            f"{self.repo_id}_world_{world_size}_rank_{rank}",
            image_writer_process=self.image_writer_process,
            image_writer_threads=self.image_writer_threads,
            push_to_hub=False,
            num_shards=world_size,
            shard_index=rank,
        )


class AggregateDatasets(PipelineStep):
    def run(self, data=None, rank: int = 0, world_size: int = 1):
        print("aggregation")


def main(slurm=True):
    # breakpoint()
    # for dir_ in Path("/fsx/remi_cadene/.cache/huggingface/lerobot/cadene").glob("droid_world*"):
    #     shutil.rmtree(dir_)

    now = dt.datetime.now()
    port_job_name = "port_openx_droid"
    logs_dir = Path("/fsx/remi_cadene/logs")
    # port_log_dir = logs_dir / f"{now:%Y-%m-%d}_{now:%H-%M-%S}_{port_job_name}"
    port_log_dir = Path("/fsx/remi_cadene/logs/2025-02-22_00-12-00_port_openx_droid")

    if slurm:
        executor_class = SlurmPipelineExecutor
        dist_extra_kwargs = {
            "job_name": port_job_name,
            "tasks": 10000,
            "workers": 20,  # 8 * 16,
            "time": "08:00:00",
            "partition": "hopper-cpu",
            "cpus_per_task": 24,
            "mem_per_cpu_gb": 2,
            # "max_array_launch_parallel": True,
        }
    else:
        executor_class = LocalPipelineExecutor
        dist_extra_kwargs = {
            "tasks": 1,
            "workers": 1,
        }

    port_executor = executor_class(
        pipeline=[
            PortOpenXDataset(
                raw_dir=Path("/fsx/mustafa_shukor/droid"), repo_id="cadene/droid_2025-02-22_00-12-00"
            ),
        ],
        logging_dir=str(port_log_dir),
        **dist_extra_kwargs,
    )
    port_executor.run()

    # if slurm:
    #     merge_extra_kwargs = {}
    # else:
    #     merge_extra_kwargs = {
    #         "job_name": "aggregate",
    #         "time": "00:01:00",
    #         "partition": "hopper-cpu",
    #     }

    # merge_executor = executor_class(
    #     depends=dist_executor,
    #     pipeline=[
    #         Aggregate(),
    #     ],
    #     logging_dir=f"/fsx/remi_cadene/logs/openx_rlds_merge",
    #     tasks=1,
    #     workers=1,
    #     **merge_extra_kwargs,
    # )
    # merge_executor.run()


if __name__ == "__main__":
    main()
