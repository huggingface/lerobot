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

"""Simulator integrations: one module per simulator, imported only when used.

Each defines ``env_fns(task, n_envs, **kwargs)``, returning ``{suite: {task_id: [factory, ...]}}``, and
``canonical_observation(obs)``, mapping observations to LeRobot keys. ``conventions(kwargs)`` is
optional: the sim's default fps, control mode, names and robot type.

A simulator that steps a whole batch at once (one GPU world with many envs) defines
``vector_env_fns(task, n_envs, **kwargs)`` instead, returning ``{suite: {task_id: factory}}``: each
factory builds one Gymnasium vector env of ``n_envs`` envs, like Gymnasium's ``vector_entry_point``. That
env must not autoreset (``metadata["autoreset_mode"]`` is ``AutoresetMode.DISABLED``), takes one seed per
env in ``reset``, returns batched observations and ``info["is_success"]`` per env, and has
``_max_episode_steps`` and ``task_description`` attributes. ``canonical_observation`` still maps one env's
observation.
"""

import contextlib
from collections.abc import Iterator
from importlib.metadata import metadata


@contextlib.contextmanager
def simulator_required(sim_type: str) -> Iterator[None]:
    """Turn a missing package in ``sim_type``'s integration into an error with install instructions.

    Wrap every call into integration code: some integrations import their simulator lazily.
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
