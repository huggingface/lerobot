#!/usr/bin/env bash
# Build (and optionally push) all lerobot benchmark eval images.
#
# Usage:
#   # Build locally only (for testing on this machine)
#   bash docker/build_benchmark_images.sh
#
#   # Build and push to Docker Hub under your org
#   bash docker/build_benchmark_images.sh --push --hub_org=pepijn223
#
#   # Force-rebuild base image (e.g. after Dockerfile.eval-base changes)
#   bash docker/build_benchmark_images.sh --no-cache-base --push --hub_org=pepijn223
#
#   # Build only specific benchmarks
#   bash docker/build_benchmark_images.sh --benchmarks="libero_plus robomme"
#
# After building, run eval with:
#   lerobot-eval --eval.runtime=docker --eval.docker.pull=false \
#     --eval.docker.image=<hub_org>/lerobot-benchmark-<benchmark>:latest ...
#   OR (if run locally with the default tag):
#   lerobot-eval --eval.runtime=docker --eval.docker.pull=false \
#     --env.type=<benchmark> ...   # auto-resolves to lerobot-benchmark-<benchmark>

set -euo pipefail

PUSH=false
HUB_ORG=""
BENCHMARKS="libero libero_plus robomme robocasa"
NO_CACHE_BASE=false
PROGRESS="auto"

for arg in "$@"; do
    case "$arg" in
        --push)            PUSH=true ;;
        --hub_org=*)       HUB_ORG="${arg#*=}" ;;
        --benchmarks=*)    BENCHMARKS="${arg#*=}" ;;
        --no-cache-base)   NO_CACHE_BASE=true ;;
        --plain)           PROGRESS="plain" ;;
        *)                 echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$PUSH" == "true" && -z "$HUB_ORG" ]]; then
    echo "ERROR: --push requires --hub_org=<your-dockerhub-org>"
    exit 1
fi

ok()   { echo "[OK]   $*"; }
fail() { echo "[FAIL] $*"; exit 1; }

BASE_CACHE_FLAG=""
if [[ "$NO_CACHE_BASE" == "true" ]]; then
    BASE_CACHE_FLAG="--no-cache"
fi

echo "=== Building lerobot-eval-base ==="
docker build \
    ${BASE_CACHE_FLAG} \
    --progress="${PROGRESS}" \
    -f "${SCRIPT_DIR}/Dockerfile.eval-base" \
    -t lerobot-eval-base:latest \
    "${REPO_ROOT}" || fail "lerobot-eval-base build failed"
ok "lerobot-eval-base"

for BENCHMARK in $BENCHMARKS; do
    LOCAL_TAG="lerobot-benchmark-${BENCHMARK}:latest"
    DOCKERFILE="${SCRIPT_DIR}/Dockerfile.eval-${BENCHMARK//_/-}"

    # Handle underscore → hyphen mapping for filename lookup
    DOCKERFILE_HYPHEN="${SCRIPT_DIR}/Dockerfile.eval-${BENCHMARK//_/-}"
    DOCKERFILE_UNDERSCORE="${SCRIPT_DIR}/Dockerfile.eval-${BENCHMARK}"
    if [[ -f "$DOCKERFILE_HYPHEN" ]]; then
        DOCKERFILE="$DOCKERFILE_HYPHEN"
    elif [[ -f "$DOCKERFILE_UNDERSCORE" ]]; then
        DOCKERFILE="$DOCKERFILE_UNDERSCORE"
    else
        fail "No Dockerfile found for benchmark '${BENCHMARK}' (tried ${DOCKERFILE_HYPHEN} and ${DOCKERFILE_UNDERSCORE})"
    fi

    echo ""
    echo "=== Building ${LOCAL_TAG} from $(basename ${DOCKERFILE}) ==="
    docker build \
        --progress="${PROGRESS}" \
        -f "${DOCKERFILE}" \
        -t "${LOCAL_TAG}" \
        "${REPO_ROOT}" || fail "${LOCAL_TAG} build failed"
    ok "${LOCAL_TAG}"

    if [[ "$PUSH" == "true" ]]; then
        HUB_TAG="${HUB_ORG}/lerobot-benchmark-${BENCHMARK}:latest"
        docker tag "${LOCAL_TAG}" "${HUB_TAG}"
        docker push "${HUB_TAG}" || fail "push ${HUB_TAG} failed"
        ok "Pushed ${HUB_TAG}"
    fi
done

echo ""
echo "=== Smoke-testing images ==="
for BENCHMARK in $BENCHMARKS; do
    LOCAL_TAG="lerobot-benchmark-${BENCHMARK}:latest"
    echo "  Smoke test: ${LOCAL_TAG}"
    docker run --rm -e BENCHMARK="${BENCHMARK}" \
        "${LOCAL_TAG}" bash docker/smoke_test_benchmark.sh \
        && ok "smoke test ${BENCHMARK}" \
        || echo "[WARN] smoke test failed for ${BENCHMARK} (may need GPU)"
done

echo ""
echo "All benchmark images built successfully."
if [[ "$PUSH" == "true" ]]; then
    echo "Pushed to Docker Hub under: ${HUB_ORG}/"
    echo ""
    echo "To use Hub images in eval, pass:"
    for BENCHMARK in $BENCHMARKS; do
        echo "  --eval.docker.image=${HUB_ORG}/lerobot-benchmark-${BENCHMARK}:latest"
    done
fi
