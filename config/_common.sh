# Shared by the teleoperation scripts. Not executable on its own.
#
# The virtualenv lives in the XLeRobot workspace, not here - see
# config/README.md. Everything runs through `uv run --project "$XLEROBOT"`
# so no activation is needed and you cannot accidentally use the wrong venv.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root (this file lives in config/)
XLEROBOT="${XLEROBOT:-$(cd "$REPO/.." && pwd)/XLeRobot}"

if [[ ! -f "$XLEROBOT/pyproject.toml" ]]; then
  echo "error: no XLeRobot workspace at $XLEROBOT" >&2
  echo "       clone it as a sibling of this repo, or set XLEROBOT=/path/to/XLeRobot" >&2
  exit 1
fi

# Two spellings of the same thing.
#
#   run some-cmd ...          a normal call; the script waits for it
#   exec "${RUN[@]}" ...      hand the process over and do not come back
#
# The array is not decoration. `exec` replaces the shell with an EXECUTABLE,
# and a shell function is not one, so the obvious-looking `exec run python`
# dies at startup with "exec: run: not found". Expanding an array in its
# place hands exec a real command.
RUN=(uv run --project "$XLEROBOT")
run() { "${RUN[@]}" "$@"; }

# Fail with something readable instead of a serial traceback. Tonight's
# lesson: a half-seated USB cable looks exactly like a config error.
require_dev() {
  local path="$1" what="$2"
  if [[ ! -e "$path" ]]; then
    echo "error: $what not found at" >&2
    echo "       $path" >&2
    echo >&2
    echo "       Present now:" >&2
    ls /dev/serial/by-id/ 2>/dev/null | sed 's/^/         /' >&2 || echo "         (none)" >&2
    echo >&2
    echo "       Check the cable is fully seated and the arm is powered." >&2
    exit 1
  fi
}

# --------------------------------------------------------------- torch
#
# On a Jetson, a bare `uv sync` quietly swaps CUDA torch for the CPU build:
# the environment is made to match the lock, the jetson extra is not in it,
# and nothing anywhere reports a problem. Every script that runs real work
# calls this so the swap is noticed the same day rather than months later.
#
# The check is a filename glob, not a python import - it has to be free
# enough to run on every script start.

is_jetson() { [[ -r /etc/nv_tegra_release ]]; }

# Echoes the installed torch version, e.g. 2.12.1+cu132 or 2.11.0
torch_version() {
  local d
  for d in "$XLEROBOT"/.venv/lib/python3*/site-packages/torch-*.dist-info; do
    [[ -d "$d" ]] || continue
    basename "$d" | sed -E 's/^torch-(.*)\.dist-info$/\1/'
    return 0
  done
  return 1
}

torch_is_cuda() { [[ "$(torch_version 2>/dev/null)" == *+cu* ]]; }

torch_report() {
  local v
  v="$(torch_version 2>/dev/null)" || { echo "torch   : not installed"; return; }
  if torch_is_cuda; then
    echo "torch   : $v  (GPU)"
  else
    echo "torch   : $v  (CPU)"
  fi
}

# Loud, but not fatal - a CPU build still teleoperates and still records.
warn_if_cpu_torch() {
  is_jetson || return 0
  torch_is_cuda && return 0
  local v; v="$(torch_version 2>/dev/null || echo '(none)')"
  echo                                                                      >&2
  echo "  ============================================================"     >&2
  echo "  CPU TORCH ON A JETSON: $v"                                        >&2
  echo "  ============================================================"     >&2
  echo "  A bare 'uv sync' replaced the CUDA build. Nothing here will"      >&2
  echo "  fail because of it - teleoperation and recording do not touch"    >&2
  echo "  CUDA - but any policy will run on the CPU and only look slow."    >&2
  echo                                                                      >&2
  echo "    ./config/sync.sh        # always right for this machine"        >&2
  echo                                                                      >&2
}
