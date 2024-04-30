# TODO(aliberts): Mute logging for these tests
import subprocess
import sys
from pathlib import Path


def _find_and_replace(text: str, finds_and_replaces: list[tuple[str, str]]) -> str:
    for f, r in finds_and_replaces:
        assert f in text
        text = text.replace(f, r)
    return text


def _run_script(path):
    subprocess.run([sys.executable, path], check=True)


def test_example_1():
    path = "examples/1_load_lerobot_dataset.py"
    _run_script(path)
    assert Path("outputs/examples/1_load_lerobot_dataset/episode_5.mp4").exists()


def test_examples_3_and_2():
    """
    Train a model with example 3, check the outputs.
    Evaluate the trained model with example 2, check the outputs.
    """

    path = "examples/3_train_policy.py"

    with open(path) as file:
        file_contents = file.read()

    # Do less steps, use smaller batch, use CPU, and don't complicate things with dataloader workers.
    file_contents = _find_and_replace(
        file_contents,
        [
            ("training_steps = 5000", "training_steps = 1"),
            ("num_workers=4", "num_workers=0"),
            ('device = torch.device("cuda")', 'device = torch.device("cpu")'),
            ("batch_size=64", "batch_size=1"),
        ],
    )

    # Pass empty globals to allow dictionary comprehension https://stackoverflow.com/a/32897127/4391249.
    exec(file_contents, {})

    for file_name in ["model.pt", "config.yaml"]:
        assert Path(f"outputs/train/example_pusht_diffusion/{file_name}").exists()

    path = "examples/2_evaluate_pretrained_policy.py"

    with open(path) as file:
        file_contents = file.read()

    # Do less evals, use CPU, and use the local model.
    file_contents = _find_and_replace(
        file_contents,
        [
            ('"eval.n_episodes=10"', '"eval.n_episodes=1"'),
            ('"eval.batch_size=10"', '"eval.batch_size=1"'),
            ('"device=cuda"', '"device=cpu"'),
            (
                '# folder = Path("outputs/train/example_pusht_diffusion")',
                'folder = Path("outputs/train/example_pusht_diffusion")',
            ),
            ('hub_id = "lerobot/diffusion_policy_pusht_image"', ""),
            ("folder = Path(snapshot_download(hub_id)", ""),
        ],
    )

    assert Path("outputs/train/example_pusht_diffusion").exists()
