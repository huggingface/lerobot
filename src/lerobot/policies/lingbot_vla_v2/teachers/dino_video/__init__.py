# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""First-party DINO-video teacher runtime.

This subpackage groups the weight-compatible DINO-video teacher designed in
``docs/source/lingbot_vla_v2_dino_video_design.md``. It never contains or
imports upstream Lumos/Meta DINOv3 source; developers only supply the published
``teacher_step_10000.pth`` + ``config.yaml`` weight files.

The only public surface is re-exported here:

- :class:`DinoVideoTeacher` — frozen teacher with ``get_future_feature``;
- :func:`build_dino_video_teacher` — builds from an ``align_params.video`` dict.

``DepthTeacherBundle`` keeps its own explicit disabled-state error until the
P4 wiring lands, so this package stays lazily imported when DINO is off.
"""

from .teacher import DinoVideoTeacher, build_dino_video_teacher

__all__ = ["DinoVideoTeacher", "build_dino_video_teacher"]
