#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scripts.backfill_pi052_action_tokenizer import (
    CHECKPOINT_DIRECTORIES,
    DEFAULT_REPOSITORIES,
    artifact_fingerprint,
    make_portable_preprocessor,
)


def test_atomic4_backfill_covers_every_repository_and_checkpoint():
    assert len(DEFAULT_REPOSITORIES) == 6
    assert CHECKPOINT_DIRECTORIES == (
        "",
        "checkpoints/003000/pretrained_model",
        "checkpoints/006000/pretrained_model",
        "checkpoints/009000/pretrained_model",
        "checkpoints/012000/pretrained_model",
    )


def test_backfill_embeds_recipe_and_declares_relative_tokenizer():
    recipe = {"messages": [{"role": "user", "content": "${task}", "stream": "low_level"}]}
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "normalizer_processor",
                "config": {},
                "state_file": "normalizer.safetensors",
            },
            {"registry_name": "render_messages_processor", "config": {"recipe": recipe}},
            {
                "registry_name": "action_tokenizer_processor",
                "config": {"action_tokenizer_name": "/fsx/original/tokenizer"},
            },
        ],
    }

    portable = make_portable_preprocessor(preprocessor)

    assert portable["steps"][1]["config"]["recipe"] == recipe
    assert portable["steps"][2]["config"]["action_tokenizer_name"] == "action_tokenizer"
    assert portable["steps"][2]["artifacts"] == {"action_tokenizer_name": "action_tokenizer"}
    assert preprocessor["steps"][2]["config"]["action_tokenizer_name"].startswith("/fsx/")


def test_artifact_fingerprint_includes_paths_and_contents():
    first = artifact_fingerprint([("a/file", b"same"), ("b/file", b"content")])

    assert first == artifact_fingerprint([("b/file", b"content"), ("a/file", b"same")])
    assert first != artifact_fingerprint([("a/renamed", b"same"), ("b/file", b"content")])
    assert first != artifact_fingerprint([("a/file", b"changed"), ("b/file", b"content")])
