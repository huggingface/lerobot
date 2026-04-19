#!/bin/bash
# Kill teleop server and release serial port connections
echo "Killing processes on port 4443 (teleop server)..."
lsof -ti :4443 | xargs kill 2>/dev/null && echo "  Killed." || echo "  None found."

echo "Killing processes using robot serial port..."
lsof -t /dev/tty.usbmodem* 2>/dev/null | xargs kill 2>/dev/null && echo "  Killed." || echo "  None found."

echo "Disabling motor torque..."
conda run -n lerobot python -c "
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
import glob

ports = glob.glob('/dev/tty.usbmodem*')
if not ports:
    print('  No robot port found.')
else:
    cfg = SO101FollowerConfig(port=ports[0], id='follower', use_degrees=True)
    robot = SO101Follower(cfg)
    robot.bus.connect()
    robot.bus.disable_torque()
    robot.bus.disconnect()
    print('  Torque disabled.')
" 2>&1 | grep -v "^WARNING\|^matplotlib\|^ERROR conda"

echo "Done."
