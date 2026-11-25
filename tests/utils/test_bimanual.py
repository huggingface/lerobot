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

import pytest

from lerobot.utils.bimanual import BimanualMixin


class FakeArm:
    def __init__(self, fail=False, fail_disconnect=False):
        self.fail = fail
        self.fail_disconnect = fail_disconnect
        self.is_connected = False

    def connect(self, calibrate=True):
        if self.fail:
            raise ConnectionError("port missing")
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False
        if self.fail_disconnect:
            raise ConnectionError("close failed")


class FakePair(BimanualMixin):
    def __init__(self, right_fails):
        self.left_arm = FakeArm()
        self.right_arm = FakeArm(fail=right_fails)


def test_failed_right_arm_disconnects_the_left_arm():
    pair = FakePair(right_fails=True)
    with pytest.raises(ConnectionError):
        pair.connect()
    assert not pair.left_arm.is_connected


def test_connects_both_arms():
    pair = FakePair(right_fails=False)
    pair.connect()
    assert pair.is_connected


def test_disconnect_reaches_the_right_arm_when_the_left_one_fails():
    pair = FakePair(right_fails=False)
    pair.connect()
    pair.left_arm.fail_disconnect = True
    with pytest.raises(ConnectionError):
        pair.disconnect()
    assert not pair.right_arm.is_connected
