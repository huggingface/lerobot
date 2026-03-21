#!/usr/bin/env bash
# Smoke-test a benchmark container: verifies imports and CLI entry-points.
#
# Run inside the container (BENCHMARK env var must be set):
#   bash docker/smoke_test_benchmark.sh
#
# Or run all benchmarks via docker compose:
#   for svc in libero libero_plus robomme robocasa; do
#     docker compose -f docker/docker-compose.benchmark.yml run --rm "$svc" \
#       bash docker/smoke_test_benchmark.sh
#   done

set -euo pipefail

BENCHMARK="${BENCHMARK:-libero}"
PASS=0
FAIL=0

ok()   { echo "[PASS] $*"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }

python_import() {
    local module="$1"
    if python -c "import ${module}" 2>/dev/null; then
        ok "import ${module}"
    else
        fail "import ${module}"
    fi
}

cli_help() {
    local cmd="$1"
    if "${cmd}" --help > /dev/null 2>&1; then
        ok "${cmd} --help"
    else
        fail "${cmd} --help"
    fi
}

echo "=== Smoke test: benchmark=${BENCHMARK} ==="

# ── lerobot core ──────────────────────────────────────────────────────────────
python_import "lerobot"
python_import "lerobot.envs"
python_import "lerobot.configs.eval"
cli_help "lerobot-eval"

# ── Benchmark-specific env import ─────────────────────────────────────────────
case "${BENCHMARK}" in
    libero)
        python_import "lerobot.envs.libero"
        python -c "
from lerobot.envs.configs import LiberoEnv
cfg = LiberoEnv(task='libero_spatial/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet')
print('  LiberoEnv config OK:', cfg.type)
" && ok "LiberoEnv config instantiation" || fail "LiberoEnv config instantiation"
        ;;

    libero_plus)
        python_import "lerobot.envs.libero"
        python -c "
from lerobot.envs.configs import LiberoPlusEnv
cfg = LiberoPlusEnv()
print('  LiberoPlusEnv config OK:', cfg.type)
" && ok "LiberoPlusEnv config instantiation" || fail "LiberoPlusEnv config instantiation"
        # Verify the LIBERO-plus package itself is importable
        python_import "libero"
        python_import "robosuite"
        ;;

    robomme)
        python_import "lerobot.envs.robomme"
        python -c "
from lerobot.envs.robomme import ROBOMME_TASKS, RoboMMEGymEnv
assert len(ROBOMME_TASKS) == 16, f'Expected 16 tasks, got {len(ROBOMME_TASKS)}'
print('  ROBOMME_TASKS OK:', ROBOMME_TASKS[:3], '...')
" && ok "RoboMME task list" || fail "RoboMME task list"
        python -c "
from lerobot.envs.configs import RoboMMEEnv
cfg = RoboMMEEnv(task='PickXtimes')
print('  RoboMMEEnv config OK:', cfg.type)
" && ok "RoboMMEEnv config instantiation" || fail "RoboMMEEnv config instantiation"
        python_import "robomme"
        ;;

    robocasa)
        python_import "lerobot.envs.robocasa"
        python -c "
from lerobot.envs.robocasa import ACTION_DIM, STATE_DIM
assert ACTION_DIM == 12, f'Expected ACTION_DIM=12, got {ACTION_DIM}'
assert STATE_DIM == 16, f'Expected STATE_DIM=16, got {STATE_DIM}'
print('  ACTION_DIM:', ACTION_DIM, '  STATE_DIM:', STATE_DIM)
" && ok "RoboCasa constants" || fail "RoboCasa constants"
        python -c "
from lerobot.envs.configs import RoboCasaEnv
cfg = RoboCasaEnv(task='PickPlaceCounterToCabinet')
print('  RoboCasaEnv config OK:', cfg.type)
" && ok "RoboCasaEnv config instantiation" || fail "RoboCasaEnv config instantiation"
        python_import "robocasa"
        python_import "robosuite"
        ;;

    *)
        echo "Unknown BENCHMARK='${BENCHMARK}'. Valid values: libero, libero_plus, robomme, robocasa"
        exit 1
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
