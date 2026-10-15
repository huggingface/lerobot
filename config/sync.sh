#!/usr/bin/env bash
# Sync the environment with the right flags for THIS machine.
#
#   ./config/sync.sh              what every machine should run
#   ./config/sync.sh --upgrade    pass anything else through to uv sync
#
# Use this instead of `uv sync` and the question of which extras to pass
# stops existing. On a Jetson it adds --extra jetson, which is what keeps
# torch on the GPU; everywhere else it syncs plainly.
#
# WHY THIS SCRIPT EXISTS
#
# `uv sync` makes the environment match the lock. On elroy a bare sync
# therefore REPLACES CUDA torch with the CPU build, silently - no error, no
# warning, and nothing downstream fails. You find out weeks later when
# inference is inexplicably slow.
#
# That cannot be prevented in configuration. uv has no environment variable
# for extras (UV_NO_DEFAULT_GROUPS exists; UV_EXTRA does not), and
# tool.uv.default-groups lives in pyproject.toml, which rosie shares - and
# the cu132 source marker is `linux aarch64`, which matches rosie too, so
# making it a default would push CUDA torch onto the Pi.
#
# So: one command that is always right, plus the loud check in _common.sh
# for when someone runs uv sync out of habit anyway.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXTRA_ARGS=()
if is_jetson; then
  EXTRA_ARGS=(--extra jetson)
  echo "machine : Jetson ($(tr -d '\0' < /proc/device-tree/model 2>/dev/null))"
  echo "extras  : jetson  (torch from the cu132 index - GPU)"
else
  echo "machine : not a Jetson"
  echo "extras  : none    (torch from PyPI - CPU)"
fi
echo "project : $XLEROBOT"
echo

cd "$XLEROBOT" || exit 1
uv sync "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" "$@" || exit $?

echo
torch_report
