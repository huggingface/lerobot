#!/bin/bash
# =============================================================================
# Prime Intellect 4090 Provisioning for drtc
# =============================================================================
#
# Provisions a 4090 instance on Prime Intellect via REST API, SSHs in,
# generates a GitHub deploy key, clones drtc, sets up a Python venv
# with deps, and installs/configures Tailscale.
#
# Prerequisites (local):
#   curl, jq, ssh
#   ~/.prime/config.json with at least api_key and ssh_key_path
#
# Usage:
#   ./scripts/provision_prime_lerobot.sh                  # create new pod + setup
#   ./scripts/provision_prime_lerobot.sh --pod-id <ID>    # resume setup on existing pod
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors / helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
for cmd in curl jq ssh ssh-keygen; do
    command -v "$cmd" >/dev/null 2>&1 || die "Required command '$cmd' not found. Please install it."
done

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
EXISTING_POD_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pod-id)
            EXISTING_POD_ID="$2"
            shift 2
            ;;
        --pod-id=*)
            EXISTING_POD_ID="${1#*=}"
            shift
            ;;
        *)
            die "Unknown argument: $1. Usage: $0 [--pod-id <POD_ID>]"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Read ~/.prime/config.json
# ---------------------------------------------------------------------------
PRIME_CONFIG="$HOME/.prime/config.json"
[ -f "$PRIME_CONFIG" ] || die "Config file not found: $PRIME_CONFIG"

API_KEY=$(jq -r '.api_key // empty' "$PRIME_CONFIG")
BASE_URL=$(jq -r '.base_url // "https://api.primeintellect.ai"' "$PRIME_CONFIG")
SSH_KEY_PATH=$(jq -r '.ssh_key_path // empty' "$PRIME_CONFIG")
DEFAULT_GPU=$(jq -r '.default_gpu // "RTX4090_24GB"' "$PRIME_CONFIG")
DEFAULT_IMAGE=$(jq -r '.default_image // "cuda_12_4_pytorch_2_4"' "$PRIME_CONFIG")
DEFAULT_DISK_SIZE=$(jq -r '.default_disk_size // 120' "$PRIME_CONFIG")
PROVIDER_TYPE=$(jq -r '.provider_type // "runpod"' "$PRIME_CONFIG")
TEAM_ID=$(jq -r '.team_id // empty' "$PRIME_CONFIG")

[ -n "$API_KEY" ]      || die "api_key missing from $PRIME_CONFIG"
[ -n "$SSH_KEY_PATH" ] || die "ssh_key_path missing from $PRIME_CONFIG"

# Expand ~ in SSH_KEY_PATH
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"
[ -f "$SSH_KEY_PATH" ] || die "SSH key not found at $SSH_KEY_PATH"

info "Config loaded from $PRIME_CONFIG"
info "  API base URL:  $BASE_URL"
info "  GPU:           $DEFAULT_GPU"
info "  Image:         $DEFAULT_IMAGE"
info "  Disk:          ${DEFAULT_DISK_SIZE}GB"
info "  Provider:      $PROVIDER_TYPE"
info "  SSH key:       $SSH_KEY_PATH"

if [ -n "$EXISTING_POD_ID" ]; then
    # -----------------------------------------------------------------------
    # Resume mode: skip creation, use existing pod
    # -----------------------------------------------------------------------
    POD_ID="$EXISTING_POD_ID"
    POD_NAME="(existing)"
    info "Resuming setup for existing pod: $POD_ID"
else
    # -----------------------------------------------------------------------
    # Query availability API for a valid cloudId + socket + provider
    # -----------------------------------------------------------------------
    info "Querying GPU availability for ${DEFAULT_GPU} ..."

    AVAIL_RESPONSE=$(curl -sS -X GET \
        "${BASE_URL}/api/v1/availability/gpus?gpu_type=${DEFAULT_GPU}&gpu_count=1&page_size=100" \
        -H "Authorization: Bearer ${API_KEY}")

    AVAIL_FILTER=$(echo "$AVAIL_RESPONSE" | jq -r \
        --arg img "$DEFAULT_IMAGE" \
        --arg prov "$PROVIDER_TYPE" \
        '[.items[] | select(.stockStatus != "Unavailable") | select(.images | index($img))] |
         ([ .[] | select(.provider == $prov) ] // .) |
         if length > 0 then .[0] else empty end')

    [ -n "$AVAIL_FILTER" ] && [ "$AVAIL_FILTER" != "null" ] || \
        die "No available ${DEFAULT_GPU} with image ${DEFAULT_IMAGE}. Raw response:\n$(echo "$AVAIL_RESPONSE" | jq .)"

    GPU_CLOUD_ID=$(echo "$AVAIL_FILTER" | jq -r '.cloudId')
    GPU_SOCKET=$(echo "$AVAIL_FILTER" | jq -r '.socket')
    RESOLVED_PROVIDER=$(echo "$AVAIL_FILTER" | jq -r '.provider')
    STOCK_STATUS=$(echo "$AVAIL_FILTER" | jq -r '.stockStatus')
    PRICE=$(echo "$AVAIL_FILTER" | jq -r '.prices.onDemand // .prices.communityPrice // "unknown"')
    DATACENTER=$(echo "$AVAIL_FILTER" | jq -r '.dataCenter // "unknown"')
    COUNTRY=$(echo "$AVAIL_FILTER" | jq -r '.country // empty')
    SECURITY=$(echo "$AVAIL_FILTER" | jq -r '.security // "unknown"')

    ok "Found available GPU"
    info "  Cloud ID:      $GPU_CLOUD_ID"
    info "  Socket:        $GPU_SOCKET"
    info "  Provider:      $RESOLVED_PROVIDER"
    info "  Datacenter:    $DATACENTER"
    info "  Security:      $SECURITY"
    info "  Stock:         $STOCK_STATUS"
    info "  Price:         \$${PRICE}/hr"

    # -------------------------------------------------------------------
    # Register SSH public key via API (or find existing)
    # -------------------------------------------------------------------
    SSH_PUB_KEY_PATH="${SSH_KEY_PATH}.pub"
    [ -f "$SSH_PUB_KEY_PATH" ] || die "SSH public key not found at $SSH_PUB_KEY_PATH"
    SSH_PUBLIC_KEY=$(cat "$SSH_PUB_KEY_PATH")

    info "Checking for existing SSH key on Prime Intellect ..."
    EXISTING_KEYS=$(curl -sS -X GET \
        "${BASE_URL}/api/v1/ssh_keys/" \
        -H "Authorization: Bearer ${API_KEY}")

    SSH_KEY_ID=$(echo "$EXISTING_KEYS" | jq -r --arg pk "$SSH_PUBLIC_KEY" '
        if type == "array" then
            [.[] | select(.publicKey == $pk)][0].id // empty
        elif .items then
            [.items[] | select(.publicKey == $pk)][0].id // empty
        else empty end')

    if [ -n "$SSH_KEY_ID" ]; then
        ok "Found existing SSH key: $SSH_KEY_ID"
    else
        info "Uploading SSH key ..."
        SSH_KEY_NAME="lerobot-provision-$(date +%Y%m%d)"
        UPLOAD_RESPONSE=$(curl -sS -X POST \
            "${BASE_URL}/api/v1/ssh_keys/" \
            -H "Authorization: Bearer ${API_KEY}" \
            -H "Content-Type: application/json" \
            -d "$(jq -n --arg name "$SSH_KEY_NAME" --arg pk "$SSH_PUBLIC_KEY" \
                '{name: $name, publicKey: $pk}')")
        SSH_KEY_ID=$(echo "$UPLOAD_RESPONSE" | jq -r '.id // empty')
        [ -n "$SSH_KEY_ID" ] || die "Failed to upload SSH key. Response:\n$UPLOAD_RESPONSE"
        ok "SSH key uploaded: $SSH_KEY_ID"
    fi

    # -------------------------------------------------------------------
    # Create pod
    # -------------------------------------------------------------------
    POD_NAME="lerobot-$(date +%Y%m%d-%H%M%S)"
    info "Creating pod '$POD_NAME' ..."

    CREATE_PAYLOAD=$(jq -n \
        --arg name "$POD_NAME" \
        --arg cloud_id "$GPU_CLOUD_ID" \
        --arg gpu_type "$DEFAULT_GPU" \
        --arg socket "$GPU_SOCKET" \
        --arg image "$DEFAULT_IMAGE" \
        --argjson disk "$DEFAULT_DISK_SIZE" \
        --arg ssh_key_id "$SSH_KEY_ID" \
        --arg datacenter_id "$DATACENTER" \
        --arg country "$COUNTRY" \
        --arg security "$SECURITY" \
        --arg provider "$RESOLVED_PROVIDER" \
        '{
            pod: {
                name: $name,
                cloudId: $cloud_id,
                gpuType: $gpu_type,
                socket: $socket,
                gpuCount: 1,
                diskSize: $disk,
                image: $image,
                sshKeyId: $ssh_key_id,
                dataCenterId: $datacenter_id,
                country: $country,
                security: $security
            },
            provider: {
                type: $provider
            }
        }')

    if [ -n "$TEAM_ID" ]; then
        CREATE_PAYLOAD=$(echo "$CREATE_PAYLOAD" | jq --arg tid "$TEAM_ID" '. + {team: {teamId: $tid}}')
    fi

    info "Payload: $(echo "$CREATE_PAYLOAD" | jq -c .)"

    CREATE_RESPONSE=$(curl -sS -X POST \
        "${BASE_URL}/api/v1/pods/" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d "$CREATE_PAYLOAD")

    POD_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id // .pod_id // empty')
    [ -n "$POD_ID" ] || die "Failed to create pod. Response:\n$CREATE_RESPONSE"
    ok "Pod created: $POD_ID"

    # -------------------------------------------------------------------
    # Poll until ACTIVE / RUNNING (timeout 10 min)
    # -------------------------------------------------------------------
    info "Waiting for pod to become ACTIVE (timeout: 10 min) ..."
    POLL_INTERVAL=10
    MAX_WAIT=600
    ELAPSED=0

    while true; do
        STATUS_RESPONSE=$(curl -sS -X GET \
            "${BASE_URL}/api/v1/pods/${POD_ID}" \
            -H "Authorization: Bearer ${API_KEY}")

        POD_STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status // empty')

        if [[ "$POD_STATUS" == "ACTIVE" || "$POD_STATUS" == "RUNNING" ]]; then
            ok "Pod is $POD_STATUS"
            break
        fi

        if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
            die "Timed out after ${MAX_WAIT}s waiting for pod to become ACTIVE (current: $POD_STATUS)"
        fi

        info "  Status: ${POD_STATUS:-unknown} (${ELAPSED}s elapsed) ..."
        sleep "$POLL_INTERVAL"
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
    done
fi

# ---------------------------------------------------------------------------
# Get SSH connection info
# ---------------------------------------------------------------------------
info "Fetching SSH connection info ..."
SSH_STATUS_RESPONSE=$(curl -sS -X GET \
    "${BASE_URL}/api/v1/pods/status?pod_ids=${POD_ID}" \
    -H "Authorization: Bearer ${API_KEY}")

SSH_CONNECTION=$(echo "$SSH_STATUS_RESPONSE" | jq -r '
    if .data then .data[0].sshConnection // .data[0].ssh_connection // empty
    elif type == "array" then .[0].sshConnection // .[0].ssh_connection // empty
    else .sshConnection // .ssh_connection // empty
    end')

[ -n "$SSH_CONNECTION" ] || die "Could not parse SSH connection from response:\n$SSH_STATUS_RESPONSE"
ok "SSH connection: $SSH_CONNECTION"

# Parse "root@1.2.3.4 -p 22" -> user, host, port
SSH_USER=$(echo "$SSH_CONNECTION" | grep -oP '^[^@]+')
SSH_HOST=$(echo "$SSH_CONNECTION" | grep -oP '(?<=@)[^\s]+')
SSH_PORT=$(echo "$SSH_CONNECTION" | grep -oP '(?<=-p\s)\d+' || echo "22")

info "  User: $SSH_USER  Host: $SSH_HOST  Port: $SSH_PORT"

# ---------------------------------------------------------------------------
# remote_exec helper
# ---------------------------------------------------------------------------
remote_exec() {
    ssh -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -p "$SSH_PORT" \
        "${SSH_USER}@${SSH_HOST}" \
        "$@"
}

# Wait a moment for SSH to be ready
info "Waiting for SSH to be ready ..."
SSH_READY=0
for i in $(seq 1 30); do
    if remote_exec "echo ok" >/dev/null 2>&1; then
        SSH_READY=1
        break
    fi
    sleep 5
done
[ "$SSH_READY" -eq 1 ] || die "SSH not reachable after 150s"
ok "SSH is ready"

# ===========================================================================
# Phase 2: Remote setup
# ===========================================================================
echo ""
echo "=============================================="
echo "  Phase 2: Remote Setup"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# 1. Generate SSH deploy key on pod
# ---------------------------------------------------------------------------
info "Generating ed25519 deploy key on pod ..."
# TODO - make email a param
remote_exec 'ssh-keygen -t ed25519 -C "vialjack@gmail.com" -f ~/.ssh/id_ed25519 -N "" <<<y >/dev/null 2>&1 || true'
DEPLOY_PUB_KEY=$(remote_exec 'cat ~/.ssh/id_ed25519.pub')
ok "Deploy key generated"

# ---------------------------------------------------------------------------
# 2. Prompt user to add deploy key
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${YELLOW}ACTION REQUIRED: Add this deploy key to drtc${NC}"
echo "=============================================="
echo ""
echo "  Public key:"
echo ""
echo "    $DEPLOY_PUB_KEY"
echo ""
echo "  Go to: https://github.com/jackvial/drtc/settings/keys"
echo "  Click 'Add deploy key', paste the key above, and save."
echo ""
read -rp "Press Enter when done ..."
echo ""

# ---------------------------------------------------------------------------
# 3. Clone repo
# ---------------------------------------------------------------------------
info "Configuring SSH for github.com (skip host key check) ..."
remote_exec 'mkdir -p ~/.ssh && cat >> ~/.ssh/config << "SSHEOF"
Host github.com
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
SSHEOF
chmod 600 ~/.ssh/config'

info "Cloning drtc ..."
remote_exec 'git clone git@github.com:jackvial/drtc.git /workspace/drtc'
ok "Repo cloned to /workspace/drtc"

# ---------------------------------------------------------------------------
# 4. Install UV
# ---------------------------------------------------------------------------
info "Installing uv ..."
remote_exec 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ok "uv installed"

# ---------------------------------------------------------------------------
# 5. Create venv
# ---------------------------------------------------------------------------
info "Creating Python 3.12 venv ..."
remote_exec 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" && cd /workspace/drtc && uv venv --python 3.12'
ok "Venv created"

# ---------------------------------------------------------------------------
# 6. Install deps
# ---------------------------------------------------------------------------
info "Installing Python dependencies (this may take a few minutes) ..."
remote_exec 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" && cd /workspace/drtc && source .venv/bin/activate && uv pip install -e ".[async,smolvla]"'
ok "Dependencies installed"

# ---------------------------------------------------------------------------
# 7. Install Tailscale
# ---------------------------------------------------------------------------
info "Installing Tailscale ..."
remote_exec 'curl -fsSL https://tailscale.com/install.sh | sh'
ok "Tailscale installed"

# ---------------------------------------------------------------------------
# 8. Tailscale auth
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${YELLOW}ACTION REQUIRED: Tailscale auth key${NC}"
echo "=============================================="
echo ""
echo "  Generate a key at: https://login.tailscale.com/admin/settings/keys"
echo ""
read -rsp "Paste your Tailscale auth key: " TAILSCALE_AUTH_KEY
echo ""
echo ""

[ -n "$TAILSCALE_AUTH_KEY" ] || die "Tailscale auth key cannot be empty"

info "Starting tailscaled ..."
remote_exec 'tailscaled --tun=userspace-networking --state=/var/lib/tailscale/tailscaled.state > /var/log/tailscaled.log 2>&1 &'
sleep 3

info "Authenticating with Tailscale ..."
remote_exec "tailscale up --auth-key=${TAILSCALE_AUTH_KEY}"
ok "Tailscale authenticated"

# ---------------------------------------------------------------------------
# 9. Print Tailscale hostname
# ---------------------------------------------------------------------------
info "Fetching Tailscale status ..."
echo ""
TS_STATUS=$(remote_exec 'tailscale status' 2>/dev/null || true)
TS_HOSTNAME=$(remote_exec 'tailscale status --self --json' 2>/dev/null | jq -r '.Self.DNSName // empty' || true)

echo "=============================================="
echo -e "${GREEN}  Provisioning complete!${NC}"
echo "=============================================="
echo ""
echo "  Pod ID:     $POD_ID"
echo "  Pod Name:   $POD_NAME"
echo "  SSH:        ssh -i $SSH_KEY_PATH -p $SSH_PORT ${SSH_USER}@${SSH_HOST}"
echo ""
if [ -n "$TS_HOSTNAME" ]; then
    echo -e "  ${CYAN}Tailscale domain: ${TS_HOSTNAME}${NC}"
else
    echo "  Tailscale status:"
    echo "    $TS_STATUS"
fi
echo ""
echo "  Next steps:"
echo "    1. Update REMOTE_SERVER_HOST in your experiment script to the Tailscale domain above"
echo "    2. Run the experiment from the robot client"
echo ""
