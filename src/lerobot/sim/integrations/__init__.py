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

"""Simulator integrations, one module per simulator, imported by name only when selected.

Each module defines ``create_envs(task, n_envs, env_cls, **kwargs)``, returning
``{suite: {task_id: env_cls([factory, ...])}}``, and ``canonical_observation(obs)``, mapping a native
observation to canonical LeRobot names. It may define ``conventions(kwargs)``: its control rate, controller,
ordered component names and simulated body, for the settings a configuration leaves out.
"""

import contextlib
from collections.abc import Iterator
from importlib.metadata import metadata


@contextlib.contextmanager
def simulator_required(sim_type: str) -> Iterator[None]:
    """Say how to get the simulator ``sim_type`` when running its integration finds a package missing.

    Some integrations import their simulator only when building an environment, so this wraps every call
    into integration code, not only its import.
    """
    try:
        yield
    except ModuleNotFoundError as error:
        if not error.name or error.name.startswith("lerobot"):
            raise
        extras = metadata("lerobot").get_all("Provides-Extra") or []
        install = (
            f"pip install 'lerobot[{sim_type}]'" if sim_type.replace("_", "-") in extras else "install it"
        )
        raise ModuleNotFoundError(
            f"The {sim_type} simulator is not installed ('{error.name}' is missing): {install}, "
            "or serve it from its Docker image with lerobot-env-server --docker",
            name=error.name,
        ) from error
