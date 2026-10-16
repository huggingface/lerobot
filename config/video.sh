#!/usr/bin/env bash
# Camera and video-link tools. Run ON THE CART.
#
#   ./config/video.sh probe /dev/cam_left [--buffersize 2]
#                         driver buffer depth: how stale a frame can be
#   ./config/video.sh view [--seconds 120] [--only left_wrist]
#                         the real cameras to the Mac's viewer, no motor bus
#                         touched; loop rate and display cost, PASS/FAIL.
#                         --help for the glass-to-glass lag test
#                         (python3 config/ms_clock.py on the Mac)
#   ./config/video.sh synthetic [<viewer-ip>] [--fps 1 --size 160 ...]
#                         a generated pattern to the viewer: separates the
#                         network and viewer from the cameras
#
# Start ./config/viewer.sh on the Mac first for view and synthetic.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_video.sh"
export PYTHONPATH="$REPO/config${PYTHONPATH:+:$PYTHONPATH}"
PY=("${RUN[@]}" python -m video_tools)

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
  probe)
    [[ $# -ge 1 ]] || { echo "usage: $0 probe /dev/cam_left [--buffersize N]" >&2; exit 2; }
    require_dev "$1" "camera"
    exec "${PY[@]}" probe "$@" ;;
  view)
    for dev in /dev/cam_left /dev/cam_right; do
      [[ -e "$dev" ]] || echo "warning: $dev missing - see ORIN-SETUP.md (camera udev rule)" >&2
    done
    cd "$REPO"
    exec "${PY[@]}" view "$WATCH_HOST" --port "$WATCH_PORT" "$@" ;;
  synthetic)
    # No arguments: the same viewer host the real runs use, so the test and
    # the thing it tests cannot drift apart.
    [[ $# -eq 0 ]] && set -- "$WATCH_HOST" --port "$WATCH_PORT"
    echo "rerun   : $(rerun_version 2>/dev/null || echo 'not installed') on this machine"
    exec "${PY[@]}" synthetic "$@" ;;
  *)
    exec "${PY[@]}" ${cmd:+"$cmd"} "$@" ;;   # help, or the unknown-command message
esac
