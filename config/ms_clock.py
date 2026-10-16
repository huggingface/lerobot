#!/usr/bin/env python3
"""A millisecond clock for the glass-to-glass lag test. Run it on the Mac.

    python3 config/ms_clock.py

Make the terminal font as large as you can (Cmd-+), point a wrist camera
at it, put the terminal beside the rerun viewer, and screenshot both at
once. Live clock minus the clock in the camera's picture = end-to-end lag.
See config/video_tools/view.py.
"""
import sys
import time

try:
    while True:
        t = time.time()
        sys.stdout.write("\r  %s.%03d  " % (time.strftime("%H:%M:%S", time.localtime(t)), int(t * 1000) % 1000))
        sys.stdout.flush()
        time.sleep(0.002)
except KeyboardInterrupt:
    print()
