# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.motors.feetech.tables import (
    STS_SMS_SERIES_BAUDRATE_TABLE,
    STS_SMS_SERIES_CONTROL_TABLE,
    STS_SMS_SERIES_ENCODINGS_TABLE,
)

# HX-30HM uses the same register layout, resolution, baudrate table,
# sign-magnitude encoding, and protocol as the Feetech STS3215.
# http://www.hiwonder.com

MODEL_CONTROL_TABLE = {
    "hx30hm": STS_SMS_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION = {
    "hx30hm": 4096,
}

MODEL_BAUDRATE_TABLE = {
    "hx30hm": STS_SMS_SERIES_BAUDRATE_TABLE,
}

MODEL_ENCODING_TABLE = {
    "hx30hm": STS_SMS_SERIES_ENCODINGS_TABLE,
}

MODEL_NUMBER_TABLE = {
    "hx30hm": 777,
}

MODEL_PROTOCOL = {
    "hx30hm": 0,
}
