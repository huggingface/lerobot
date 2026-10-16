#!/usr/bin/env bash
# Read a recorded dataset back: frames, features, decoded camera frames as
# PNGs, video files. Local only. See config/dataset_check.py.
#
#   ./config/dataset-check.sh carlkesselman/xlerobot-test    newest of that name
#
# lerobot-record appends _YYYYMMDD_HHMMSS to every new dataset's name; a
# plain name picks the newest. Frames land in ./readback/.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "${RUN[@]}" python "$REPO/config/dataset_check.py" "$@"
