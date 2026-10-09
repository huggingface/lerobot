#!/usr/bin/env bash
# Prove the video link on its own, with no robot attached.
#
#   ./config/video-test.sh                 # the default viewer host
#   ./config/video-test.sh 192.168.1.52
#   ./config/video-test.sh --local         # this machine's desktop
#
# See config/video_test.py for what it is for and what to expect.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_video.sh"

args=("$@")
# No host and not --local: fall back to the same host the real runs use, so
# the test and the thing it is testing cannot drift apart.
if [[ $# -eq 0 ]]; then
  args=("$WATCH_HOST" --port "$WATCH_PORT")
fi

echo "rerun   : $(rerun_version 2>/dev/null || echo 'not installed') on this machine"
exec "${RUN[@]}" python "$REPO/config/video_test.py" "${args[@]}"
