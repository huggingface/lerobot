#!/usr/bin/env bash
# The real cameras to the rerun viewer, with no robot - arms can be unpowered.
# Measures what video costs the loop, and sets up the glass-to-glass lag test.
#
#   ./config/cam-view.sh                    all three cameras, 60 s, to the Mac
#   ./config/cam-view.sh --seconds 300
#   ./config/cam-view.sh --only left_wrist
#   ./config/cam-view.sh --image-fps all    every tick, not --watch's 10 Hz
#   ./config/cam-view.sh --help             the lag test, step by step
#
# Start ./config/viewer.sh on the Mac first. See config/cam_view.py.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_video.sh"

for dev in /dev/cam_left /dev/cam_right; do
  [[ -e "$dev" ]] || echo "warning: $dev missing - see ORIN-SETUP.md (camera udev rule)" >&2
done

cd "$REPO"
exec "${RUN[@]}" python "$REPO/config/cam_view.py" "$WATCH_HOST" --port "$WATCH_PORT" "$@"
