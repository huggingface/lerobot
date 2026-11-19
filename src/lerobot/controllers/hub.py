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

"""Loading controller checkpoints from the Hugging Face Hub."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)


def load_onnx_policy(
    repo_id: str, filename: str, metadata_keys: Sequence[str] = ()
) -> tuple[ort.InferenceSession, dict[str, np.ndarray]]:
    """Download an ONNX policy from the Hub and open it, with the constants baked into its metadata.

    Args:
        repo_id: Hub repository holding the policy.
        filename: Path of the ONNX file in the repository.
        metadata_keys: Metadata entries to read, each a JSON list of floats (PD gains, default pose, ...).

    Returns:
        The inference session, and each requested metadata entry as a float32 array.
    """
    import onnxruntime as ort

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    # Controllers are small MLPs stepped from a background thread next to the torch policy. One
    # thread is the lowest latency, and an ORT pool spanning every core starves both loops.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=options)
    logger.info(
        f"Loaded {repo_id}/{filename}: {session.get_inputs()[0].shape} → {session.get_outputs()[0].shape}"
    )
    if not metadata_keys:
        return session, {}

    import onnx

    metadata = {prop.key: prop.value for prop in onnx.load(path, load_external_data=False).metadata_props}
    missing = [key for key in metadata_keys if key not in metadata]
    if missing:
        raise ValueError(
            f"{repo_id}/{filename} must contain {list(metadata_keys)} in its ONNX metadata (missing {missing})"
        )
    return session, {key: np.array(json.loads(metadata[key]), dtype=np.float32) for key in metadata_keys}
