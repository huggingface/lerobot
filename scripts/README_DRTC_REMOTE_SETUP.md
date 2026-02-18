# DRTC Remote Setup

- Provision 4090 instance on prime intellect
- Run drex prime setup 0
- drex prime ssh 0 and run generate ssh key pair `ssh-keygen -t ed25519 -C "vialjack@gmail.com"`
- Add public key to repo deploy keys
- Clone lerobot-jack
- setup venv `uv venv --python 3.12`
- install deps `uv pip install -e ."[async, smolvla]"
- Install tailscale on the node `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up --auth-key=yourauthkey`
- `tailscaled --tun=userspace-networking --state=/var/lib/tailscale/tailscaled.state &`
- `tailscale up`
- Update run_drtc_experiment_with_remote_server.sh to point at this node
- run ./scripts/run_drtc_experiment_with_remote_server.sh on your client 
