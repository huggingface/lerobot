#!/usr/bin/env python

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

import threading

import torch

from lerobot.policies.common.export_utils import is_non_strict_exporting


# The output shows which branch the check chose when the module was traced or run.
class Branch(torch.nn.Module):
    def forward(self, x):
        return x + 1 if is_non_strict_exporting() else x - 1


def test_is_non_strict_exporting_reads_true_only_inside_non_strict_export():
    x = torch.zeros(1)
    assert torch.equal(Branch()(x), x - 1)
    assert torch.equal(torch.export.export(Branch(), (x,), strict=False).module()(x), x + 1)


def test_is_non_strict_exporting_reads_false_on_other_threads_during_an_export():
    paused, resume = threading.Event(), threading.Event()
    errors = []

    class Paused(torch.nn.Module):
        def forward(self, x):
            paused.set()
            resume.wait(timeout=60)
            return x + 1

    def export():
        try:
            torch.export.export(Paused(), (torch.zeros(1),), strict=False)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=export)
    thread.start()
    try:
        assert paused.wait(timeout=60), errors
        # PyTorch's export flag is set for the whole process, but this thread is not exporting.
        assert torch.compiler.is_exporting() and not is_non_strict_exporting()
    finally:
        resume.set()
        thread.join(timeout=60)
    assert not errors


def test_is_non_strict_exporting_stays_off_under_torch_compile_and_strict_export():
    x = torch.zeros(1)
    # Dynamo picks the branch while it traces, so the eager backend shows what the check read.
    assert torch.equal(torch.compile(Branch(), backend="eager", fullgraph=True)(x), x - 1)
    assert torch.equal(torch.export.export(Branch(), (x,), strict=True).module()(x), x - 1)
