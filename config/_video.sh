# Shared video handling for the cart-side scripts. Sourced, not run.
#
# The cart has the cameras; the operator is at the Pi. lerobot's display
# layer can send either way round, so both are offered:
#
#   --watch            push to a rerun viewer, by default on the Mac. You
#                      get lerobot's full blueprint - a panel per camera
#                      plus action and observation time series - rendered
#                      wherever the viewer runs.
#
#                      START THE VIEWER FIRST; the cart connects out to it
#                      and init_rerun runs during startup, so nothing
#                      listening means the launch fails rather than
#                      carrying on without video.
#
#                        uvx --from rerun-sdk rerun --port 9876
#
#                      Override with WATCH_HOST=<ip>. If the cart cannot
#                      connect, the viewer may be bound to localhost only -
#                      check `rerun --help` for a bind option.
#
#   --watch=web        serve Foxglove from the cart on :8765 and open a
#                      browser on the Pi. No viewer install needed, but it
#                      is a websocket out of the recording process and
#                      heavier on the Pi.
#
#   --watch=local      render here, on the cart's own desktop. Only useful
#                      with a monitor plugged into the Orin.
#
# Images are JPEG-compressed before they go on the wire in the remote
# cases. These frames are for the operator, not the dataset - the dataset
# is written from the uncompressed originals on this machine - so costing
# them some quality to keep the link responsive is free.

# The Mac, not rosie. The operator station is headless by design - it reads
# two serial buses and a HID device and needs no display - and the video
# originates on the cart anyway, so the viewer can be any machine on the
# network. The Mac is the better screen and does not compete with the
# leader host for the Pi's CPU during a recording run.
WATCH_HOST="${WATCH_HOST:-192.168.1.52}"    # the Mac
WATCH_PORT="${WATCH_PORT:-9876}"

# JPEG-compress before anything crosses the wire, in every remote mode.
# These frames are for the operator; the dataset is written from the
# uncompressed originals on the cart, so quality spent here is free.
#
# It also matters for STABILITY, not just bandwidth. rerun's sink
# back-pressures: when it cannot keep up - or cannot connect at all - its
# batcher channel fills and the sender BLOCKS INSIDE THE CONTROL LOOP:
#
#   WARN re_quota_channel::sync: batcher_output: Sender has been blocked
#   for over 5 seconds waiting for space in channel
#
# Measured: a run pointed at a host with no viewer listening sat at 22 Hz
# where the same configuration without video managed 29.
DISPLAY_ARGS=()

# Scans ALL arguments and removes the one it consumes, so --watch can sit
# anywhere. Sets DISPLAY_ARGS and rewrites the caller's positional
# parameters via REMAINING.
parse_watch_all() {
  REMAINING=()
  local a found=1
  for a in "$@"; do
    case "$a" in
      --watch|--watch=web|--watch=local) parse_watch "$a" && found=0 ;;
      *) REMAINING+=("$a") ;;
    esac
  done
  return $found
}

parse_watch() {
  case "${1:-}" in
    --watch)
      DISPLAY_ARGS=(--display_data=true --display_mode=rerun
                    --display_ip="$WATCH_HOST" --display_port="$WATCH_PORT"
                    --display_compressed_images=true)
      echo "video   : pushing to the rerun viewer at $WATCH_HOST:$WATCH_PORT"
      echo "          START THE VIEWER FIRST. With nothing listening, rerun"
      echo "          does not just fail - it back-pressures into the control"
      echo "          loop and costs you several Hz."
      if command -v nc >/dev/null 2>&1 && ! nc -z -w2 "$WATCH_HOST" "$WATCH_PORT" 2>/dev/null; then
        echo
        echo "  WARNING: nothing is listening on $WATCH_HOST:$WATCH_PORT." >&2
        echo "           Start it with:  uvx --from rerun-sdk rerun --port $WATCH_PORT" >&2
        echo "           Or run without --watch." >&2
        echo
      fi
      return 0 ;;
    --watch=web)
      DISPLAY_ARGS=(--display_data=true --display_mode=foxglove
                    --display_ip=0.0.0.0 --display_port=8765
                    --display_compressed_images=true)
      echo "video   : Foxglove at ws://$(hostname).local:8765 - open it on $WATCH_HOST"
      return 0 ;;
    --watch=local)
      if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        echo "error: --watch=local needs a desktop session on this machine." >&2
        echo "       Use --watch to push to the operator station instead." >&2
        exit 1
      fi
      DISPLAY_ARGS=(--display_data=true --display_mode=rerun)
      echo "video   : rerun, on this machine's desktop"
      return 0 ;;
  esac
  return 1
}
