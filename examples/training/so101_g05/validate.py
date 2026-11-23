"""Check actual G05 CoT targets and token limits without allocating policy weights."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.language_render import render_sample
from lerobot.datasets.recipe import TrainingRecipe
from lerobot.policies.g05.configuration_g05 import G05Config
from lerobot.policies.g05.processor_g05 import _apply_recipe_cot
from lerobot.policies.g05.scratchpad_g05 import Scratchpad, runtime_command
from lerobot.policies.g05.tokenizer_g05 import G05Tokenizer


def validate(root, checkpoint, max_episodes=0):
    config = G05Config.from_pretrained(checkpoint)
    recipe = TrainingRecipe.from_dict(config.recipe)
    tokenizer = G05Tokenizer(checkpoint / "hf_processor", config.author_model_config)
    seen = set()
    counts = Counter()
    maxima = Counter()

    def check(persistent, events, timestamp, task, branch):
        rendered = render_sample(
            recipe=recipe.blend[branch],
            persistent=persistent,
            events=events,
            t=timestamp,
            sample_idx=0,
            task=task,
        )
        sample = {}
        _apply_recipe_cot(config, False, sample, rendered, 0, 1)
        command = rendered["messages_rendered"][0]["content"]
        size = len(tokenizer.encode_text(command))
        limit = 200 if branch == "atomic_task" else 768
        if size > limit:
            raise ValueError(f"{branch} input needs {size} tokens, exceeds {limit}")
        maxima["input_tokens"] = max(maxima["input_tokens"], size)
        if branch == "scratchpad_cot":
            text = "|".join([v for k in ("bbox", "atomic_task") if (v := sample.get(k))]) + "|Action: <EOV>"
            size = len(tokenizer.encode_text(text))
            maxima["generated_tokens"] = max(maxima["generated_tokens"], size)
            if size > 768:
                raise ValueError(f"CoT needs {size} output tokens; exceeds 768")
            state = Scratchpad.parse(sample["atomic_task"].removeprefix("Subtask: "))
            counts["boxes"] += int("bbox" in sample)
            counts["grasp_points"] += bool(state.grasp_point)
            # Inference receives only previous predictions, not the current targets.
            empty = runtime_command(config.recipe, task, Scratchpad())
            if "<previous_memory></previous_memory>" not in empty:
                raise ValueError("Initial runtime prompt does not match empty-memory training context")
        counts[branch] += 1

    stop = False
    for path in sorted((root / "data").rglob("*.parquet")):
        if stop:
            break
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=128, columns=["episode_index", "language_persistent", "language_events", "timestamp"]
        ):
            for row in batch.to_pylist():
                ep = row["episode_index"]
                if ep not in seen and max_episodes and len(seen) >= max_episodes:
                    stop = True
                    break
                persistent = row["language_persistent"]
                tasks = [r["content"] for r in persistent if r["style"] == "task_aug"]
                task = max(tasks, key=len) if tasks else "Execute the task."
                if ep not in seen:
                    seen.add(ep)
                    subtasks = [r for r in persistent if r["style"] == "subtask"]
                    if not subtasks or subtasks[-1]["content"] != "done":
                        raise ValueError(f"Episode {ep} needs prepared terminal labels")
                    for subtask in subtasks:
                        for branch in ("scratchpad_cot", "atomic_task"):
                            check(persistent, [], subtask["timestamp"], task, branch)
                if row["language_events"]:
                    check(persistent, row["language_events"], row["timestamp"], task, "scratchpad_cot")
            if stop:
                break
    if not seen or not counts["boxes"] or not counts["grasp_points"]:
        raise ValueError(f"Missing required episode/box/grasp supervision: {counts}")
    return {
        "episodes": len(seen),
        "full_dataset": not bool(max_episodes),
        "coverage": dict(counts),
        "maxima": dict(maxima),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.checkpoint, args.max_episodes), indent=2))


if __name__ == "__main__":
    main()
