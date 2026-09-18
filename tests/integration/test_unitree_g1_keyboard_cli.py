"""Real terminal input and CLI parsing; no X server or physical hardware."""

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.getenv("G1_KEYBOARD_HUB_TESTS"), reason="Requires Hub model, DDS, and POSIX PTY")
def test_headless_keyboard_cli(tmp_path):
    import pty

    master, slave = pty.openpty()
    env = dict(os.environ, MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE"):
        env.pop(key, None)
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_teleoperate",
        "--robot.type=unitree_g1",
        "--robot.is_simulation=true",
        "--robot.sim_publish_images=false",
        "--robot.sim_onscreen=false",
        "--robot.cameras={}",
        "--teleop.type=unitree_g1_keyboard",
        "--display_data=false",
        "--teleop_time_s=12",
    ]
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, env=env)
    os.close(slave)
    output = ""
    sent = False
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    output += os.read(master, 65536).decode(errors="replace")
                except OSError:
                    break
            if not sent and "G1 keyboard ready:" in output:
                os.write(master, b"\nl4+")
                sent = True
            if sent and "G1 keyboard jog: kLeftElbow.q=" in output:
                os.write(master, b" ")
                break
            if process.poll() is not None:
                break
        assert sent and "G1 keyboard jog: kLeftElbow.q=" in output, output
        # Continue draining the PTY so loop telemetry cannot block shutdown.
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    output += os.read(master, 65536).decode(errors="replace")
                except OSError:
                    break
        assert process.wait(timeout=5) == 0, output
        assert "measured-pose hold" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        (Path(tmp_path) / "keyboard-cli.log").write_text(output)
