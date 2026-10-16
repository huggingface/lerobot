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

"""Convert trusted LIBERO initial-state files during image build, before removing torch."""

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    count = 0
    for path in args.root.rglob("*.pruned_init"):
        value = torch.load(path, weights_only=False)  # nosec B614: trusted image-build assets
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Invalid initial states: {path}")
        np.save(path.with_suffix(path.suffix + ".npy"), array, allow_pickle=False)
        count += 1
    if not count:
        raise ValueError(f"No LIBERO initial-state assets found: {args.root}")
    print(f"Converted {count} initial-state assets")


if __name__ == "__main__":
    main()
