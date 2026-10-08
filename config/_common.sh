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

run() { uv run --project "$XLEROBOT" "$@"; }

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
