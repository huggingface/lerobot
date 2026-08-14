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

"""Package glue for the ``patrol`` example strategy — see ``patrol.py`` for the example.

Why this example is a package rather than a single module, i.e. the two mechanics that a
third-party strategy has to satisfy:

1. ``--strategy.discover_packages_path=<path>`` imports *a package* and then iterates its
   submodules (``load_plugin`` in ``lerobot/configs/parser.py`` reads ``__path__``), so
   pointing it at a plain ``.py`` module raises ``PluginLoadError: ... it is a module, not
   a package. discover_packages_path needs a package (a directory with an __init__.py).``
2. ``create_strategy`` resolves the implementation class by naming convention
   (``make_device_from_device_class`` in ``lerobot/utils/import_utils.py``): it strips the
   trailing ``Config`` from the config class name and imports ``PatrolStrategy`` from the
   config module's **parent** package — here ``examples.rollout.custom_strategy``, i.e.
   this file — or from ``<parent>.patrolstrategy``.  The config's *own* module is never
   searched, so the re-export below is what makes the lookup succeed.  Drop it and the
   registered config still parses, but ``create_strategy`` fails with ``ImportError: Could
   not locate device class 'PatrolStrategy'``, naming both modules it tried.

Importing this package also registers the strategy: the ``@register_subclass("patrol")``
decorator runs as a side effect of the import below, before draccus parses the CLI.
"""

from .patrol import PatrolStrategy, PatrolStrategyConfig

__all__ = ["PatrolStrategy", "PatrolStrategyConfig"]
