# Microduck connection setup

LeRobot connects to the Duck's `robotd` service over SSH. You can let LeRobot manage the
connection automatically or keep a manual tunnel open. The camera connects directly to the Duck
with WebRTC; it does not travel through this SSH tunnel.

Use a charged battery and put your laptop and Duck on a network that allows them to reach each other.
Find the Duck's current IP with `duckctl ip`, then set it in each terminal you use:

```bash
export DUCK_IP=192.168.1.42 # replace with your Duck's IP
```

## Set up SSH access once

First connect normally and verify the host key before accepting it:

```bash
ssh "microduck@$DUCK_IP"
```

Enter the Duck's SSH password when prompted. Run `exit` to return to your laptop.

Automatic connection uses SSH keys and cannot prompt for a password. If you do not already have
an SSH key, create one on your laptop with `ssh-keygen -t ed25519`. Keep an existing key if prompted;
do not overwrite it. Then install its public key on the Duck:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub "microduck@$DUCK_IP"
```

This asks for the Duck's password once. For a passphrase-protected key, load it into your SSH agent:

```bash
ssh-add ~/.ssh/id_ed25519
```

Check that a non-interactive connection succeeds:

```bash
ssh -o BatchMode=yes "microduck@$DUCK_IP" 'echo Connection_OK'
```

It should print `Connection_OK` without asking for a password. SSH credentials are separate from your
Hugging Face token. Never copy a private key to the Duck; `ssh-copy-id` installs the public key only.

## Automatic connection

Supply the IP and omit `robot.robotd_address`. LeRobot creates a private local socket, opens the SSH
forward, and removes its own connection on exit. No second terminal is needed.

For a live view while controlling the Duck with its paired gamepad:

```bash
# Start padd if it was stopped for keyboard teleoperation.
ssh "microduck@$DUCK_IP" sudo systemctl start padd

uv run --no-sync lerobot-teleoperate \
    --robot.type=microduck --robot.remote_ip="$DUCK_IP" \
    --robot.control_mode=gamepad --teleop.type=microduck_gamepad \
    --fps=15 --teleop_time_s=60 --display_data=true
```

Close the Duck's browser camera console and other camera viewers first. Enable and drive the Duck
normally with the gamepad. In this mode LeRobot observes commands; it does not enable, move, stop,
or relax the Duck. Ending the LeRobot process leaves the gamepad in control.

The same robot and teleoperator flags work with `lerobot-record`. See the
[Microduck guide](../../docs/source/microduck.mdx) for the full recording command. Add
`--dataset.push_to_hub=false` to keep recordings local.

Optional connection settings are `--robot.ssh_user`, `--robot.ssh_port`, and
`--robot.ssh_timeout_s`. Host-key verification remains enabled.

## Manual tunnel

A manual tunnel also works, including when you prefer to enter an SSH password each time.
In terminal 1, run this command and leave it running:

```bash
ssh -N -o ExitOnForwardFailure=yes \
    -L 127.0.0.1:17000:/run/robotd.sock "microduck@$DUCK_IP"
```

No output after authentication is normal. Port 17000 is bound only on your laptop's loopback
interface. It forwards to the Duck's Unix socket `/run/robotd.sock`.

In terminal 2, set `DUCK_IP` again, then run:

```bash
uv run --no-sync lerobot-teleoperate \
    --robot.type=microduck --robot.remote_ip="$DUCK_IP" \
    --robot.robotd_address=127.0.0.1:17000 \
    --robot.control_mode=gamepad --teleop.type=microduck_gamepad \
    --fps=15 --teleop_time_s=60 --display_data=true
```

LeRobot uses the supplied forward and leaves it running on exit. Once finished, stop the tunnel
with Ctrl+C in terminal 1. If port 17000 is occupied, choose another port in both commands.

For keyboard or policy control, stop `padd` first and use `--robot.control_mode=lerobot` so only one
source sends movement commands:

```bash
ssh "microduck@$DUCK_IP" sudo systemctl stop padd
```

Restore the paired gamepad afterwards with `sudo systemctl start padd` over SSH.

## Troubleshooting

- **`Permission denied (publickey,password)` during automatic connection:** install the public key,
  load a passphrase-protected key into the agent, and retry the `BatchMode` check above. Alternatively,
  use a manual tunnel and set `robot.robotd_address` explicitly.
- **`Host key verification failed`:** connect with normal SSH and verify the Duck's identity. If a
  previously trusted key changed, establish why it changed before replacing the saved key.
- **Connection timeout:** check the battery, current IP, Wi-Fi connection, and whether the network
  isolates clients. Test `ssh "microduck@$DUCK_IP"` first.
- **`Address already in use`:** another process owns the manual tunnel's local port. Choose a free port
  and update `robot.robotd_address` to match.
- **Forward opens but robotd cannot be reached:** check `ssh "microduck@$DUCK_IP" systemctl status robotd`
  and the user's access to `/run/robotd.sock`.
- **Camera handshake timeout:** SSH working does not prove the camera service is reachable. Close
  competing viewers and run `uv run --no-sync python examples/microduck/check_camera.py --host "$DUCK_IP" --seconds 10`.
