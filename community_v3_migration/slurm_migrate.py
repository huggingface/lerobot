"""SLURM-distributed driver for run_migration.py.

Fans the ``community_dataset_v3`` -> v3.0 migration out across SLURM workers, mirroring the
datatrove pattern used by ``examples/dataset/slurm_recompute_stats.py``. This is a *map-only*
job: each worker owns a stride ``subs[rank::world_size]`` of the sub-datasets and runs the
exact same per-dataset pipeline as ``run_migration.py`` (download -> fix -> v2.1->v3.0 convert
-> upload -> cleanup). There is no aggregate step.

Resume is twofold and free: (1) each worker skips any sub-dataset already present in the
destination repo (``already_done``), and (2) datatrove skips ranks whose completion marker
exists. Re-run the identical command to mop up failures.

Example (numeric smoke test on one namespace, no SLURM):
    python slurm_migrate.py --slurm 0 --workers 1 \
        --dst-repo HuggingFaceVLA/community_dataset_v3_degrees \
        --work-dir ./cdv3_work --manifest-dir ./cdv3_manifests \
        --folder-name Beegbrain

Full run on the cluster:
    python slurm_migrate.py \
        --dst-repo HuggingFaceVLA/community_dataset_v3_degrees \
        --work-dir /fsx/$USER/cdv3_work \
        --manifest-dir /fsx/$USER/cdv3_manifests \
        --logs-dir /fsx/$USER/logs/cdv3_migrate \
        --workers 64 --partition hopper-cpu --qos normal \
        --cpus-per-task 4 --mem-per-cpu 4G \
        --env-command "source /fsx/$USER/venvs/lerobot/bin/activate; export HF_TOKEN=<token>"

IMPORTANT: workers must reach the internet (HF download + upload) and have a write-scoped
HF token (HF_TOKEN) in --env-command. Keep --workers modest (many concurrent commits to one
destination repo contend); rely on resume passes to clear transient upload failures.
"""

import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

MIGRATION_DIR = str(Path(__file__).resolve().parent)


class MigrateShard(PipelineStep):
    """Each worker migrates its ``subs[rank::world_size]`` slice of sub-datasets."""

    def __init__(
        self,
        subs,
        dst_repo,
        work_dir,
        manifest_dir,
        migration_dir,
        no_push=False,
        only_classify=False,
        standalone=False,
    ):
        super().__init__()
        self.subs = subs
        self.dst_repo = dst_repo
        self.work_dir = work_dir
        self.manifest_dir = manifest_dir
        self.migration_dir = migration_dir
        self.no_push = no_push
        self.only_classify = only_classify
        self.standalone = standalone

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        # Pickled onto the worker: keep self-contained. The migration package dir must be on
        # sys.path so ``run_migration`` and its siblings (classify/fix_dataset/so_arm_frame)
        # import.
        import csv
        import logging
        import shutil
        import sys
        import time
        import traceback
        from pathlib import Path

        if self.migration_dir not in sys.path:
            sys.path.insert(0, self.migration_dir)

        from classify import classify
        from huggingface_hub import HfApi
        from run_migration import already_done, download_subfolder, migrate_one

        from lerobot.utils.utils import init_logging

        init_logging()

        my_subs = self.subs[rank::world_size]
        if not my_subs:
            logging.info(f"Rank {rank}: no sub-datasets assigned")
            return
        logging.info(f"Rank {rank}: {len(my_subs)} / {len(self.subs)} sub-datasets")

        # Per-rank scratch and manifest so workers never collide (migrate_one wipes
        # work_dir/<namespace> around each dataset).
        work_dir = str(Path(self.work_dir) / f"rank_{rank:05d}")
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.manifest_dir).mkdir(parents=True, exist_ok=True)
        manifest = Path(self.manifest_dir) / f"manifest_{rank:05d}.csv"

        api = HfApi()
        # Resume prefetch. A single transient Hub read timeout here must NOT kill the whole
        # rank (and skip its entire dataset slice), so retry with backoff and, as a last
        # resort, fall back to an empty set (already-present datasets are re-checked per item
        # and, for --source molmoact, were already filtered out on the submit node).
        dst_files: set = set()
        if not self.only_classify and not self.no_push:
            for attempt in range(5):
                try:
                    dst_files = set(api.list_repo_files(self.dst_repo, repo_type="dataset"))
                    break
                except Exception as e:
                    if attempt == 4:
                        logging.warning(f"Rank {rank}: could not list {self.dst_repo} after 5 "
                                        f"tries ({e}); proceeding without a resume set.")
                    else:
                        time.sleep(5 * (attempt + 1))

        fieldnames = sorted(
            {"root", "robot_type", "is_so", "encoding", "action_dim", "maxabs", "ambiguous", "action"}
        )
        write_header = not manifest.exists()
        with open(manifest, "a", newline="") as mf:
            w = csv.DictWriter(mf, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for i, sub in enumerate(my_subs):
                try:
                    if self.only_classify:
                        if self.standalone:
                            from huggingface_hub import snapshot_download

                            snapshot_download(repo_id=sub, repo_type="dataset",
                                              local_dir=str(Path(work_dir) / sub),
                                              allow_patterns=["meta/*"])
                        else:
                            download_subfolder(sub, work_dir, patterns=[f"{sub}/meta/*"])
                        row = {"root": sub, **classify(Path(work_dir) / sub)}
                        shutil.rmtree(Path(work_dir) / sub.split("/")[0], ignore_errors=True)
                    elif not self.no_push and already_done(api, self.dst_repo, sub, dst_files):
                        row = {"root": sub, "action": "skipped: already present in destination repo"}
                    else:
                        row = migrate_one(api, self.dst_repo, sub, work_dir, self.no_push,
                                          standalone=self.standalone)
                except Exception as e:
                    row = {"root": sub, "action": f"ERROR: {e}"}
                    traceback.print_exc()
                w.writerow({k: row.get(k) for k in fieldnames})
                mf.flush()
                logging.info(f"Rank {rank} [{i + 1}/{len(my_subs)}] {sub}: {row.get('action')}")


def _mem_gb(mem: str) -> int:
    s = str(mem).strip().lower().rstrip("b").rstrip("g")
    return int(float(s))


def _make_executor(pipeline, logs_dir, job_name, slurm, workers, time, partition, cpus, mem, qos, env_command, venv_path):
    kwargs = {"pipeline": pipeline, "logging_dir": str(Path(logs_dir) / job_name)}
    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,
                "workers": workers,
                "time": time,
                "partition": partition,
                "cpus_per_task": cpus,
                "mem_per_cpu_gb": _mem_gb(mem),
                "sbatch_args": {},
            }
        )
        if qos:
            kwargs["qos"] = qos
        if venv_path:
            kwargs["venv_path"] = venv_path
        if env_command:
            kwargs["env_command"] = env_command
        return SlurmPipelineExecutor(**kwargs)
    kwargs.update({"tasks": workers, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def main():
    import sys

    if MIGRATION_DIR not in sys.path:
        sys.path.insert(0, MIGRATION_DIR)
    from huggingface_hub import HfApi
    from migrate_molmoact import REFERENCE_REPO, list_molmoact_datasets, pending_datasets
    from run_migration import SRC_REPO, list_datasets, resolve_folders

    p = argparse.ArgumentParser(
        description="SLURM-distributed migration to LeRobotDataset v3.0 (map-only). Source is "
                    "either the community_dataset_v3 monorepo or the standalone datasets listed "
                    "by allenai/MolmoAct2-SO100_101-Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", choices=("monorepo", "molmoact"), default="monorepo",
                   help="'monorepo': HuggingFaceVLA/community_dataset_v3 subfolders. "
                        "'molmoact': standalone datasets listed by allenai/MolmoAct2-SO100_101-Dataset.")
    p.add_argument("--reference-repo", default=REFERENCE_REPO, metavar="ORG/NAME",
                   help="(--source molmoact) Repo whose datasets are already migrated and skipped.")
    p.add_argument("--dst-repo", default=None, metavar="ORG/NAME", help="Destination HF dataset repo.")
    p.add_argument("--work-dir", default="./cdv3_work", help="Scratch root; each rank gets a subdir.")
    p.add_argument("--manifest-dir", default="./cdv3_manifests", help="Per-rank manifest CSVs land here.")
    p.add_argument("--logs-dir", type=Path, default=Path("logs"), help="datatrove logs dir.")
    p.add_argument("--job-name", default="cdv3_migrate", help="SLURM job name.")
    p.add_argument("--workers", type=int, default=64, help="Number of parallel SLURM tasks.")
    p.add_argument("--slurm", type=int, default=1, help="1 = submit via SLURM; 0 = run locally.")
    p.add_argument("--partition", default=None, help="SLURM partition, e.g. 'hopper-cpu'.")
    p.add_argument("--qos", default=None, help="SLURM QoS, e.g. 'normal'.")
    p.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per SLURM task.")
    p.add_argument("--mem-per-cpu", default="4G", help="Memory per CPU, e.g. '4G'.")
    p.add_argument("--time", default="24:00:00", help="Wall-clock limit per task.")
    p.add_argument("--venv-path", default=None, help="venv activate script sourced on each worker.")
    p.add_argument("--env-command", default=None, help="Raw shell snippet run before python (export HF_TOKEN, etc.).")
    p.add_argument("--folder-name", nargs="+", default=None, help="Target specific folders/namespaces instead of all.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N sub-datasets (ignored with --folder-name).")
    p.add_argument("--only-classify", action="store_true", help="Only classify + write manifest; no convert/upload.")
    p.add_argument("--no-push", action="store_true", help="Fix + convert locally, keep output, do not upload.")
    args = p.parse_args()

    if not args.no_push and not args.only_classify and not args.dst_repo:
        p.error("--dst-repo is required unless --no-push or --only-classify is set.")

    api = HfApi()
    standalone = args.source == "molmoact"
    if standalone:
        all_ids = list_molmoact_datasets(api)
        if args.folder_name:
            wanted = {n.strip("/") for n in args.folder_name}
            subs = [s for s in all_ids if s in wanted]
        else:
            subs = pending_datasets(api, all_ids, args.dst_repo, args.reference_repo,
                                    args.no_push, args.only_classify)
            if args.limit:
                subs = subs[: args.limit]
    elif args.folder_name:
        subs = resolve_folders(api, SRC_REPO, args.folder_name)
    else:
        subs = list_datasets(api, SRC_REPO)
        if args.limit:
            subs = subs[: args.limit]
    print(f"{len(subs)} sub-datasets targeted", file=sys.stderr)
    if not subs:
        p.error("no sub-datasets resolved")

    # Create the destination repo once on the submit node so workers don't race on it.
    if not args.only_classify and not args.no_push:
        api.create_repo(args.dst_repo, repo_type="dataset", exist_ok=True)

    executor = _make_executor(
        pipeline=[
            MigrateShard(
                subs,
                args.dst_repo,
                args.work_dir,
                args.manifest_dir,
                MIGRATION_DIR,
                no_push=args.no_push,
                only_classify=args.only_classify,
                standalone=standalone,
            )
        ],
        logs_dir=args.logs_dir,
        job_name=args.job_name,
        slurm=args.slurm == 1,
        workers=args.workers,
        time=args.time,
        partition=args.partition,
        cpus=args.cpus_per_task,
        mem=args.mem_per_cpu,
        qos=args.qos,
        env_command=args.env_command,
        venv_path=args.venv_path,
    )
    executor.run()


if __name__ == "__main__":
    main()
