#!/usr/bin/env python3
"""
Check if the Unitree locomotion DDS domain / LocoClient is already initialized.
Does NOT send any commands — safe to run.
"""

import sys
import psutil  # pip install psutil if missing
import os

sys.path.append('/Users/nepyope/Documents/unitree/unitree_IL_lerobot/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelFactory

def is_loco_domain_active():
    """Detect if a ChannelFactory singleton is already initialized in this process."""
    fac = ChannelFactory()
    active = getattr(fac, "_ChannelFactory__initialized", False)
    participant = getattr(fac, "_ChannelFactory__participant", None)
    domain = getattr(fac, "_ChannelFactory__domain", None)
    return active, participant, domain

def check_other_python_processes():
    """See if another Python process using the Unitree SDK is alive."""
    procs = []
    for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            if "python" in p.info["name"].lower() and any("unitree" in arg for arg in p.info["cmdline"]):
                procs.append((p.info["pid"], " ".join(p.info["cmdline"])))
        except Exception:
            pass
    return procs

if __name__ == "__main__":
    active, participant, domain = is_loco_domain_active()
    print("=== Local DDS Factory State ===")
    print(f"Factory initialized: {active}")
    print(f"Participant: {participant}")
    print(f"Domain: {domain}")
    print()

    print("=== Other Python processes using Unitree SDK ===")
    procs = check_other_python_processes()
    if not procs:
        print("No other active Unitree-related Python processes found.")
    else:
        for pid, cmd in procs:
            print(f"PID {pid}: {cmd}")

    print()
    if os.path.exists("/tmp/CycloneDDS"):
        print("Warning: /tmp/CycloneDDS exists — could indicate leftover DDS shared memory.")
