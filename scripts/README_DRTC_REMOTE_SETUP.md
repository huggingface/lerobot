# DRTC Remote Setup

- Provision and set up prime intellect gpu instance:
  - `./scripts/provision_prime_lerobot.sh --deploy-key-email your.email@example.com`
- Add the generated deploy public key to your repository deploy keys in GitHub.
- Install and authenticate Tailscale on the remote node, then copy the node DNS name.
- Run experiments from the robot client with a required remote host flag:
  - `./scripts/run_drtc_experiment_with_remote_server.sh --remote-server-host <TAILSCALE_DOMAIN> --config mixture_of_faults`
- If you are tunneling over SSH, start the client with an explicit tunnel target:
  - `./scripts/start_drtc_client.sh --tunnel-ssh-user-host root@<REMOTE_HOST>`
