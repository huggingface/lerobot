"""Create a separate local initialization config around the proven SO-101 G05 weights."""

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from lerobot.datasets.recipe import TrainingRecipe
from lerobot.policies.g05.configuration_g05 import G05Config, make_g05_cot_prompt_template


def prepare(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    recipe = TrainingRecipe.from_yaml(Path(__file__).with_name("recipe.yaml"))
    config = G05Config.from_pretrained(source)
    config.device = "cuda"
    config.recipe = asdict(recipe)
    config.recipe_path = None
    config.memory_scratchpad = True
    config.use_language_recipe = True
    config.preserve_pretrained_processors = True
    config.runtime_system = "system2"
    config.predict_cot = True
    config.runtime_cot_fields = ("bbox", "subtask")
    config.cot_bbox_camera = "observation.images.front"
    config.prompt_template = make_g05_cot_prompt_template(
        config.num_prompt_images,
        fields=config.runtime_cot_fields,
        flow_only="<action_action" not in config.prompt_template,
    ).replace("<command_text_!_200>", "<command_text_!_768>")
    config.author_model_config["ar"]["max_new_tokens"] = 768
    config.author_model_config["fm"]["num_flow_samples"] = 1
    config.optimizer_lr = 2e-5
    config.scheduler_warmup_steps = 500
    config.push_to_hub = False
    config.private = False
    config.repo_id = "maximellerbach/so101-g05-full-v1"
    # The copied sidecars and weights belong to the new local initialization directory.
    config.pretrained_path = output
    output.mkdir(parents=True)
    for path in source.rglob("*"):
        if not path.is_file() or ".cache" in path.parts or path.name == "config.json":
            continue
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".safetensors":
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)
        else:
            shutil.copy2(path, destination)
    config.save_pretrained(output)
    (output / "recipe.yaml").write_text(Path(__file__).with_name("recipe.yaml").read_text())
    (output / "recipe_provenance.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "repo": "maximellerbach/g05_so101_sim4k",
                "revision": "b10e721b763a655e2df7188e86991502cb583d88",
                "subfolder": "checkpoints/004000/pretrained_model",
                "dataset": "maximellerbach/so101-fineart-mixed-v1",
                "dataset_revision": "6cd89066f9768dbee8c868af4fa8f108de189089",
                "action_calibration": "unchanged pretrained processors; no new dataset normalization",
            },
            indent=2,
        )
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/maxime/github/robots/models/g05_so101_sim4k/checkpoints/004000/pretrained_model"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
    print(args.output)


if __name__ == "__main__":
    main()
