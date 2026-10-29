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

"""The env server: serves a sim to other processes or machines.

``lerobot-env-server`` steps a ``LocalVectorEnv`` for each session, and ``RemoteVectorEnv`` drives it with
the same Gymnasium API, over the transport remote inference uses. Simulator images import it without torch.
"""
