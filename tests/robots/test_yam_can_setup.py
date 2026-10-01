# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy
from unittest.mock import Mock

import pytest

from examples.yam import prepare_can as setup
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import YamArmConfig


def link(up=True):
    return {
        "flags": ["UP"] if up else [],
        "mtu": 16,
        "linkinfo": {
            "info_kind": "can",
            "info_data": {"state": "ERROR-ACTIVE", "bittiming": {"bitrate": 1_000_000}},
        },
    }


@pytest.fixture
def env(monkeypatch):
    states = {"can0": link(), "can1": link()}
    writes = []

    def run(command, check):
        assert check
        writes.append(command)
        if command[-1] == "up":
            states[command[-2]] = link()

    monkeypatch.setattr(setup.shutil, "which", lambda _: "/usr/sbin/ip")
    monkeypatch.setattr(setup.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(setup, "verify_adapter", Mock())
    monkeypatch.setattr(setup, "_link", lambda ip, port: deepcopy(states[port]))
    monkeypatch.setattr(setup.subprocess, "run", run)
    arms = [YamArmConfig(port=port, expected_adapter_serial=port + "-serial") for port in states]
    return states, writes, arms


def test_up_interfaces_are_unchanged(env):
    _, writes, arms = env
    setup.prepare_can(*arms)
    assert writes == []


def test_down_interface_enabled_at_classic_one_megabit(env):
    states, writes, arms = env
    states["can0"] = link(up=False)
    setup.prepare_can(*arms)
    assert writes == [
        [
            "sudo",
            "-n",
            "/usr/sbin/ip",
            "link",
            "set",
            "can0",
            "type",
            "can",
            "bitrate",
            "1000000",
            "fd",
            "off",
        ],
        ["sudo", "-n", "/usr/sbin/ip", "link", "set", "can0", "up"],
    ]


def test_wrong_second_adapter_prevents_any_changes(env, monkeypatch):
    states, writes, arms = env
    states["can0"] = link(up=False)
    monkeypatch.setattr(setup, "verify_adapter", Mock(side_effect=[None, ValueError("wrong serial")]))
    with pytest.raises(ValueError, match="wrong serial"):
        setup.prepare_can(*arms)
    assert writes == []


@pytest.mark.parametrize("bad", ["BUS-OFF", "ERROR-PASSIVE", "wrong-bitrate", "can-fd"])
def test_bad_active_interface_is_never_reset(env, bad):
    states, writes, arms = env
    states["can0"] = link(up=False)
    data = states["can1"]["linkinfo"]["info_data"]
    if bad == "wrong-bitrate":
        data["bittiming"]["bitrate"] = 500_000
    elif bad == "can-fd":
        states["can1"]["mtu"] = 72
    else:
        data["state"] = bad
    with pytest.raises(ValueError, match="not resetting"):
        setup.prepare_can(*arms)
    assert writes == []
