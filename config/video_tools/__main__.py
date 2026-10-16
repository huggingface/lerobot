"""./config/video.sh <command> [args]  -  camera and video-link tools.

    probe       one camera's driver buffer depth and hidden latency (no lerobot)
    view        the real cameras to the rerun viewer, no robot; loop and display cost
    synthetic   a generated test pattern to the viewer: the network and viewer alone

./config/video.sh <command> --help for each.
"""

import importlib
import sys

COMMANDS = {"probe": "probe", "view": "view", "synthetic": "synthetic"}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0 if len(sys.argv) >= 2 else 2
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"unknown command '{cmd}'. One of: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"video_tools.{COMMANDS[cmd]}")
    sys.argv = [f"video.sh {cmd}", *sys.argv[2:]]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
