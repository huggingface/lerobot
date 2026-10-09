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
#                        ./config/viewer.sh      # on the Mac
#
#                      which reads the version out of uv.lock, because the
#                      viewer and the SDK here are two halves of one rerun
#                      release. By hand it is
#
#                        uvx --from rerun-sdk==<version> rerun
#
#                      with NO other arguments. That one process both opens
#                      the window and listens on 0.0.0.0:9876 for the cart -
#                      it is not a viewer that needs a server put in front
#                      of it. `--port 9876` is the default and adds nothing.
#
#                      On startup it says "Listening for gRPC connections
#                      ... Connect by running `rerun --connect ...`". That
#                      is printed by every mode that starts the server,
#                      window or no window, and is only relevant if you
#                      want a SECOND viewer attached from elsewhere. It is
#                      not an instruction. Ignore it.
#
#                      Override the host with WATCH_HOST=<ip>. If the cart
#                      cannot reach it, suspect the viewer machine's
#                      firewall before anything else - rerun already binds
#                      0.0.0.0, but macOS blocks incoming connections to new
#                      binaries by default and a uvx-launched rerun is a new
#                      binary every time the cache moves.
#
#                      To test the link with no robot in the way:
#                        ./config/video-test.sh
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
      # The operator view does not run at the control rate, and should not.
      #
      # At the full 30 Hz the cart pushes three 640x480 JPEGs a frame - a few
      # MB/s up the radio, plus the encoding - and that starves the thread
      # receiving leader actions on the same machine. Measured: with --watch
      # the link stalled for ~320 ms about once a second, base stopping and
      # arms holding each time; without --watch, not once in the same run.
      #
      # 10 Hz is unmistakably live to drive by and leaves the actions alone.
      # Raise it if your link is better than ours; 0 sends scalars only.
      export LEROBOT_RERUN_IMAGE_FPS="${LEROBOT_RERUN_IMAGE_FPS:-10}"
      DISPLAY_ARGS=(--display_data=true --display_mode=rerun
                    --display_ip="$WATCH_HOST" --display_port="$WATCH_PORT"
                    --display_compressed_images=true)
      local rv; rv="$(rerun_version 2>/dev/null || echo unknown)"
      echo "video   : pushing to the rerun viewer at $WATCH_HOST:$WATCH_PORT"
      echo "          image rate $LEROBOT_RERUN_IMAGE_FPS Hz (control loop stays at the configured fps)"
      echo "          this machine has rerun-sdk $rv - the viewer must match."
      echo "          On the Mac:  ./config/viewer.sh   (reads uv.lock)"
      echo "          By hand:     uvx --from rerun-sdk==$rv rerun"
      echo "          Never bare 'uvx --from rerun-sdk rerun' - that is"
      echo "          PyPI latest, which is not what this machine sends."
      echo "          START IT FIRST. With nothing there,"
      echo "          rerun does not just fail - it back-pressures into the"
      echo "          control loop and costs you several Hz."
      if command -v nc >/dev/null 2>&1 && ! nc -z -w2 "$WATCH_HOST" "$WATCH_PORT" 2>/dev/null; then
        echo
        echo "  WARNING: nothing answering on $WATCH_HOST:$WATCH_PORT." >&2
        echo "           Start it on the Mac:  ./config/viewer.sh" >&2
        echo "           Already running? Then it is reachability, not rerun:" >&2
        echo "           on macOS, System Settings > Network > Firewall blocks" >&2
        echo "           incoming connections to new binaries by default." >&2
        echo "           Isolate it with:   ./config/video-test.sh" >&2
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
