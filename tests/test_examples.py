import pytest
from pathlib import Path

@pytest.mark.skip(reason="For some reason 1_visualize_dataset.py downloads the dataset")
@pytest.mark.parametrize(
    "path",
    [
        "examples/1_visualize_dataset.py",
        "examples/2_evaluate_pretrained_policy.py",
        "examples/3_train_policy.py",
    ],
)
def test_example(path):

    with open(path, 'r') as file:
        file_contents = file.read()
    exec(file_contents)

    assert Path("outputs/visualize_dataset/example/episode_0.mp4").exists()
