from pathlib import Path


def _find_and_replace(text: str, finds: list[str], replaces: list[str]) -> str:
    for f, r in zip(finds, replaces):
        assert f in text
        text = text.replace(f, r)
    return text


def test_example_1():
    path = "examples/1_visualize_dataset.py"

    with open(path, "r") as file:
        file_contents = file.read()
    exec(file_contents)

    assert Path("outputs/visualize_dataset/example/episode_0.mp4").exists()


def test_examples_3_and_2():
    """
    Train a model with example 3, check the outputs.
    Evaluate the trained model with example 2, check the outputs.
    """

    path = "examples/3_train_policy.py"

    with open(path, "r") as file:
        file_contents = file.read()

    # Do less steps and use CPU.
    file_contents = _find_and_replace(
        file_contents,
        ['"offline_steps=5000"', '"device=cuda"'],
        ['"offline_steps=1"', '"device=cpu"'],
    )

    exec(file_contents)

    for file_name in ["model.pt", "stats.pth", "config.yaml"]:
        assert Path(f"outputs/train/example_pusht_diffusion/{file_name}").exists()

    path = "examples/2_evaluate_pretrained_policy.py"

    with open(path, "r") as file:
        file_contents = file.read()

    # Do less evals and use CPU.
    file_contents = _find_and_replace(
        file_contents,
        ['"eval_episodes=10"', '"rollout_batch_size=10"', '"device=cuda"'],
        ['"eval_episodes=1"', '"rollout_batch_size=1"','"device=cpu"'],
    )

    assert Path(f"outputs/train/example_pusht_diffusion").exists()