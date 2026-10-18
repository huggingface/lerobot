<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Long-WAM reference core

This initial port preserves the computational implementation from
NVlabs/LongLive, revision `1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9`,
under `Long-WAM/src/longwam`. Original file-level copyright and provenance
notices are retained.

- NVIDIA-authored Long-WAM code and LeRobot integration additions use Apache-2.0.
- FastWAM-derived portions retain their MIT license and copyright notices.
- Incorporated DiffSynth-Studio and Wan portions retain Apache-2.0 and their
  original contributor/Alibaba copyright notices.
- Existing Hugging Face copyright notices in LeRobot are unchanged.

NVIDIA notices cover NVIDIA contributions, not ownership of third-party code.
An SPDX `MIT AND Apache-2.0` expression identifies the combined obligations,
not a choice between licenses. License texts are in `licenses/` beside this notice.

Package imports and logging were adapted for LeRobot. Formatting and typing
were modernized; mutable default lists now initialize per call and existing
zip truncation is explicit. Cache and initialization preconditions fail explicitly.
Local checkpoint loading uses `weights_only=True`. Hardware-specific configuration
entry points are excluded from the initial reference port.

External checkpoints, datasets, and simulator assets are not bundled here and
remain subject to their respective terms. Attribution does not imply endorsement.
