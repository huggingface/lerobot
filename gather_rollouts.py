import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def run_task(t):
    print(f"Starting task {t}")
    # Simulate progress monitoring
    for i in range(5):  # Assuming the task has 5 steps
        time.sleep(1)  # Simulate time taken for each step
        print(f"Task {t} progress: Step {i+1}/5")
    
    result = subprocess.run([
        "python", "lerobot/scripts/collect_rollouts.py",
        "-p", "/home/jhseon/projects/lerobot/outputs/train/2024-11-15/13-07-57_pushany_diffusion_2024-11-15_13-07-57_pushany_diffusion_woT/checkpoints/150000/pretrained_model",
        "-o", "/data/pushany_rollouts_as_hdf5",
        "-n", "5000",
        "-t", str(t)
    ], capture_output=True, text=True)
    print(f"Task {t} completed with return code {result.returncode}")
    return result

def main():
    total_tasks = 21
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = {executor.submit(run_task, t): t for t in range(total_tasks)}
        for future in as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
                print(f"Task {t} output: {result.stdout}")
            except Exception as e:
                print(f"Task {t} generated an exception: {e}")

if __name__ == "__main__":
    main()