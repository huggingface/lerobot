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

import subprocess
import sys


def test_pi052_config_import_does_not_load_model_or_dataset_processor():
    code = """
import sys
from lerobot.policies import PI052Config
assert PI052Config.__name__ == "PI052Config"
assert "lerobot.policies.pi052.modeling_pi052" not in sys.modules
assert "lerobot.policies.pi052.processor_pi052" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
