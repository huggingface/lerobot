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

"""Remove LIBERO's eager training imports and use converted initial-state assets.

This narrowly patches the pinned simulator distribution inside its image. The original
checkpoint/training environments are untouched. All initial-state loading becomes NumPy.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    benchmarks = list(args.root.rglob("libero/benchmark/__init__.py"))
    if not benchmarks:
        raise ValueError("Pinned LIBERO benchmark module was not found")
    for path in benchmarks:
        source = path.read_text()
        if "import torch" not in source:
            raise ValueError(f"Expected pinned LIBERO torch import: {path}")
        source = source.replace("import torch", "import numpy as np")
        source = source.replace("torch.load(", "_load_numpy_states(")
        source += '\n\ndef _load_numpy_states(path, *args, **kwargs):\n    return np.load(str(path) + ".npy", allow_pickle=False)\n'
        path.write_text(source)
    for path in args.root.rglob("libero/utils/utils.py"):
        source = path.read_text()
        if "torch." in source:
            raise ValueError(f"Unexpected LIBERO training dependency in runtime utility: {path}")
        path.write_text(source.replace("import torch\n", ""))


if __name__ == "__main__":
    main()
