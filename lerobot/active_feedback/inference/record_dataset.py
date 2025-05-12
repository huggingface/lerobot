import json
from pathlib import Path
from lerobot.lga.inference.run_inference import LGARunner

def main(
    goal: str,
    episode_name: str = "episode_1",
    debug: bool = False,
    device: str = "cuda",
    fps: int = 30,
):
    # 1) Init runner & LGAInference (which also instantiates TaskSuccessEvaluator)
    runner = LGARunner(debug=debug, device=device)
    runner._initialize_lga(fps=fps)

    # 2) Execute one full inference pass
    runner.run_inference(goal=goal, fps=fps)

    # 3) Collect all ChatGPT interactions
    policy_interactions = runner.selector.get_interactions()
    task_interactions   = runner.lga.task_evaluator.get_interactions()

    # 4) Combine and save
    out_dir = Path("local_dataset/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{episode_name}.json"
    with open(log_path, "w") as f:
        json.dump({
            "policy_selection": policy_interactions,
            "task_evaluation":  task_interactions,
        }, f, indent=4)

    print(f"✅ Saved policy ({len(policy_interactions)} calls) + task ({len(task_interactions)} calls) to {log_path}")

if __name__ == "__main__":
    goal = "Place a box with a blue cube in it into a bin."
    main(goal)
