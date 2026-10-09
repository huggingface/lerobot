#!/usr/bin/env bash
# Start the rerun viewer, at the version the cart is actually sending.
#
# RUN THIS ON THE MACHINE WITH THE SCREEN - the Mac, not elroy or rosie.
#
#   ./config/viewer.sh
#
# Why this exists rather than a command you type:
#
# `uvx --from rerun-sdk rerun` means "whatever PyPI calls latest today".
# elroy means "whatever uv.lock says", because its SDK comes in through
# lerobot[viz]. Those two agree only by coincidence, and right now they do
# not: the lock holds 0.33.1 and PyPI's latest is five minor releases past
# it. A rerun viewer and a rerun SDK are two halves of one release and the
# wire format moves between them, so a mismatched pair gives you a window
# that opens, sits on the welcome screen, and never says why.
#
# uv.lock is the one file both machines already agree on, and this reads
# the version straight out of it. Nothing to type, nothing to keep in sync,
# and it follows the lock automatically the next time it moves.
#
# It needs no virtualenv and no XLeRobot workspace - just this checkout and
# uv - because the viewer is not part of the robot environment.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$REPO/uv.lock"
PORT="${WATCH_PORT:-9876}"
# Cap what the viewer will hold. This is NOT optional for a long run.
#
# lerobot computes a memory limit in init_rerun and then uses it only on the
# path where it spawns the viewer itself:
#
#     memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
#     if ip and port:
#         rr.connect_grpc(url=...)        # <- limit never applied
#     else:
#         rr.spawn(memory_limit=memory_limit)
#
# We are the ip-and-port case, so nothing on the cart can cap this process.
# It has to be set here. And it matters more than it looks: camera frames
# are logged with static=True (rerun_visualization.py:175), static entities
# are not on the timeline, and the viewer evicts oldest-BY-TIME - so static
# data is never a candidate for eviction. Three cameras at 30 Hz is ninety
# such logs a second.
MEM="${VIEWER_MEMORY:-25%}"

if [[ ! -f "$LOCK" ]]; then
  echo "error: no uv.lock at $LOCK" >&2
  exit 1
fi

# The [[package]] block for rerun-sdk, then the version line under it.
VER="$(awk '
  /^name = "rerun-sdk"$/ { want = 1; next }
  want && /^version = / { gsub(/[", ]/, "", $0); sub(/^version=/, "", $0); print; exit }
' "$LOCK")"

if [[ -z "$VER" ]]; then
  echo "error: no rerun-sdk in $LOCK" >&2
  echo "       is lerobot[viz] still in the dependencies?" >&2
  exit 1
fi

if ! command -v uvx >/dev/null 2>&1; then
  echo "error: uvx not found. Install uv: https://docs.astral.sh/uv/" >&2
  exit 1
fi

cat <<TXT
viewer  : rerun $VER  (from uv.lock - the same version elroy sends)
port    : $PORT on all interfaces
memory  : $MEM  (set VIEWER_MEMORY to change)

A window will open and stay on the welcome screen until the cart connects.
That is correct; start the cart afterwards.

The "Listening for gRPC connections ... Connect by running \`rerun
--connect ...\`" line below is normal and is not telling you to do
anything. Every mode that starts the server prints it.

FIRST LAUNCH OF A VERSION IS SLOW - give it a minute or two before you
decide it is wedged. The wheel is ~125 MB, uvx fetches and unpacks it,
macOS then scans the unpacked binary because nothing signed it, and wgpu
compiles its Metal pipelines. The window appears partway through all that,
so macOS may well call it "not responding" for a while. It is cached per
version, so this happens once.

If the cart cannot reach this machine, suspect this machine's firewall
first - rerun binds 0.0.0.0 already. On macOS: System Settings > Network >
Firewall. Allow the connection when asked, or turn it off long enough to
find out whether that was the answer.

TXT

# No other arguments. Plain `rerun` is BOTH the window and the listener on
# 0.0.0.0:$PORT - there is no separate server to put in front of it, and
# --port is only here because it may have been overridden above. The mode
# to avoid is --serve-grpc, which is a server with no window.
exec uvx --from "rerun-sdk==$VER" rerun --port "$PORT" --memory-limit "$MEM"
