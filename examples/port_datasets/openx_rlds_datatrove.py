import datetime as dt
import logging
import os
import random
import time
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from huggingface_hub import CommitOperationAdd, HfApi, create_commit, preupload_lfs_files
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.utils import HfHubHTTPError

from lerobot.common.datasets.aggregate import aggregate_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import create_lerobot_dataset_card

BASE_DELAY = 0.1
MAX_RETRIES = 12


class PortOpenXDataset(PipelineStep):
    def __init__(
        self,
        raw_dir: Path | str,
        repo_id: str = None,
        image_writer_process: int = 0,
        image_writer_threads: int = 8,
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.repo_id = repo_id
        self.image_writer_process = image_writer_process
        self.image_writer_threads = image_writer_threads

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from datasets.utils.tqdm import disable_progress_bars

        from examples.port_datasets.openx_rlds import create_lerobot_dataset
        from examples.port_datasets.openx_utils.test import display_slurm_info, display_system_info
        from lerobot.common.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

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
    def __init__(
        self,
        repo_ids: list[str],
        aggregated_repo_id: str,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.aggregated_repo_id = aggregated_repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        aggregate_datasets(self.repo_ids, self.aggregated_repo_id)


class UploadDataset(PipelineStep):
    def __init__(
        self,
        repo_id: str,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        private: bool = False,
        **card_kwargs,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.branch = branch
        self.tags = tags
        self.license = license
        self.private = private
        self.card_kwargs = card_kwargs

        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "1":
            logging.warning(
                'HF_HUB_ENABLE_HF_TRANSFER is not set to "1". Install hf_transfer and set the env '
                "variable for faster uploads:\npip install hf-transfer\nexport HF_HUB_ENABLE_HF_TRANSFER=1"
            )

        self._repo_init = False

    def _create_repo(self, hub_api):
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=self.private,
            repo_type="dataset",
            exist_ok=True,
        )
        if self.branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=self.branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=self.branch):
            card = create_lerobot_dataset_card(
                tags=self.tags, dataset_info=self.meta.info, license=license, **self.card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=self.branch)

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from lerobot.common.utils.utils import init_logging

        init_logging()

        meta = LeRobotDatasetMetadata(self.repo_id)

        # TODO: list files, shard files, upload meta data for rank=0
        filenames = []

        raise NotImplementedError()

        hub_api = HfApi()
        if not self._repo_init:
            self._create_repo(hub_api)
            self._repo_init = True

        additions = [
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=meta.root / filename)
            for filename in filenames
        ]
        logging.info(f"Uploading {','.join(filenames)} to the hub...")
        preupload_lfs_files(
            repo_id=self.repo_id, repo_type="dataset", additions=additions, revision=self.revision
        )
        logging.info(f"Upload of {','.join(filenames)} to the hub complete!")
        # if self.cleanup:
        #     for filename in filenames:
        #         self.local_working_dir.rm(filename)
        self.operations.extend(additions)

    def close(self, rank: int = 0):
        filelist = list(self.output_mg.get_open_files().keys())
        super().close()
        if filelist:
            logging.info(f"Starting upload of {len(filelist)} files to {self.dataset}")
            self.upload_files(*filelist)
        retries = 0
        while True:
            try:
                create_commit(
                    self.repo_id,
                    repo_type="dataset",
                    operations=self.operations,
                    commit_message=f"DataTrove upload ({len(self.operations)} files)",
                    revision=self.revision,
                )
                break
            except HfHubHTTPError as e:
                if "A commit has happened since" in e.server_message:
                    if retries >= MAX_RETRIES:
                        logging.error(f"Failed to create commit after {MAX_RETRIES=}. Giving up.")
                        raise e
                    logging.info("Commit creation race condition issue. Waiting...")
                    time.sleep(BASE_DELAY * 2**retries + random.uniform(0, 2))
                    retries += 1
                else:
                    raise e


def make_port_executor(raw_dir, repo_id, port_job_name, port_log_dir, slurm=True):
    kwargs = {
        "pipeline": [
            PortOpenXDataset(raw_dir, repo_id),
        ],
        "logging_dir": str(port_log_dir),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": port_job_name,
                "tasks": 2048,
                "workers": 20,
                "time": "08:00:00",
                "partition": "hopper-cpu",
                "cpus_per_task": 24,
                "mem_per_cpu_gb": 2,
                "max_array_launch_parallel": True,
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update(
            {
                "tasks": 1,
                "workers": 1,
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def make_aggregate_executor(
    repo_ids, aggr_repo_id, port_job_name, aggregate_log_dir, depends=None, slurm=True
):
    kwargs = {
        "pipeline": [
            AggregateDatasets(repo_ids, aggr_repo_id),
        ],
        "logging_dir": str(aggregate_log_dir),
        "tasks": 1,
        "workers": 1,
    }
    if depends:
        kwargs["depends"] = depends

    if slurm:
        kwargs.update(
            {
                "job_name": port_job_name,
                "time": "08:00:00",
                "partition": "hopper-cpu",
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def make_upload_executor(repo_id, upload_job_name, upload_log_dir, depends=None, slurm=True):
    kwargs = {
        "pipeline": [
            UploadDataset(repo_id),
        ],
        "logging_dir": str(upload_log_dir),
        "tasks": 1,
        "workers": 1,
    }
    if depends:
        kwargs["depends"] = depends

    if slurm:
        kwargs.update(
            {
                "job_name": upload_job_name,
                "time": "08:00:00",
                "partition": "hopper-cpu",
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def main(slurm=True):
    # breakpoint()
    # for dir_ in Path("/fsx/remi_cadene/.cache/huggingface/lerobot/cadene").glob("droid_world*"):
    #     shutil.rmtree(dir_)

    world = 2048
    raw_dir = "/fsx/mustafa_shukor/droid"
    port_job_name = "port_openx_droid"
    aggregate_job_name = "aggregate_openx_droid"
    upload_job_name = "upload_openx_droid"
    logs_dir = Path("/fsx/remi_cadene/logs")
    repo_id = "cadene/droid"

    now = dt.datetime.now()
    datetime = f"{now:%Y-%m-%d}_{now:%H-%M-%S}"
    # datetime = "2025-02-22_11-17-00"

    port_log_dir = logs_dir / f"{datetime}_{port_job_name}"
    aggregate_log_dir = logs_dir / f"{datetime}_{aggregate_job_name}"
    upload_log_dir = logs_dir / f"{datetime}_{upload_job_name}"

    port_executor = make_port_executor(raw_dir, repo_id, port_job_name, port_log_dir, slurm)
    port_executor.run()

    repo_ids = [f"{repo_id}_{datetime}_world_{world}_rank_{rank}" for rank in range(world)]
    aggregate_executor = make_aggregate_executor(
        repo_ids, repo_id, aggregate_job_name, aggregate_log_dir, port_executor, slurm
    )
    aggregate_executor.run()

    upload_executor = make_upload_executor(
        repo_id, upload_job_name, upload_log_dir, aggregate_executor, slurm
    )
    upload_executor.run()


if __name__ == "__main__":
    main()
