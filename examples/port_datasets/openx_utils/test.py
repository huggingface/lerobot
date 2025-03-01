import os

import psutil


def display_system_info():
    # Get the number of CPUs
    num_cpus = psutil.cpu_count(logical=True)
    print(f"Number of CPUs: {num_cpus}")

    # Get memory information
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024**3)  # Convert bytes to GB
    available_memory = memory_info.available / (1024**3)  # Convert bytes to GB
    used_memory = memory_info.used / (1024**3)  # Convert bytes to GB

    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Available Memory: {available_memory:.2f} GB")
    print(f"Used Memory: {used_memory:.2f} GB")


def display_slurm_info():
    # Get SLURM job ID
    job_id = os.getenv("SLURM_JOB_ID")
    print(f"SLURM Job ID: {job_id}")

    # Get SLURM job name
    job_name = os.getenv("SLURM_JOB_NAME")
    print(f"SLURM Job Name: {job_name}")

    # Get the number of tasks
    num_tasks = os.getenv("SLURM_NTASKS")
    print(f"Number of Tasks: {num_tasks}")

    # Get the number of nodes
    num_nodes = os.getenv("SLURM_NNODES")
    print(f"Number of Nodes: {num_nodes}")

    # Get the number of CPUs per task
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    print(f"CPUs per Task: {cpus_per_task}")

    # Get the node list
    node_list = os.getenv("SLURM_NODELIST")
    print(f"Node List: {node_list}")

    # Get the task ID (only available within an srun task)
    task_id = os.getenv("SLURM_PROCID")
    print(f"Task ID: {task_id}")


if __name__ == "__main__":
    display_system_info()
    display_slurm_info()
