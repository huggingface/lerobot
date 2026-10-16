#!/usr/bin/env python3
"""Read every configuration and status register on one bus, side by side.

    ./config/servo.sh regs            bus 2 (right arm)
    ./config/servo.sh regs --bus 1    bus 1 (left arm + head)

Read-only. Nothing is written, torque is not touched.

`servo.sh scan` asks one question - does an id answer a ping? - and a servo can
answer perfectly while refusing to move: torque off, a torque limit of 0
left behind by its overload protection, position limits collapsed onto each
other by a corrupted EEPROM write during a brownout, an error bit in Status.
Every servo on an arm is configured the same way except for its id and
homing offset, so the odd one out is the finding: any register where one
servo differs from the majority is flagged.
"""

import argparse
import sys
from collections import Counter

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .hw import PORTS
NAMES = {
    1: ["l_pan", "l_lift", "l_elbow", "l_wflex", "l_wroll", "l_grip", "head_1", "head_2"],
    2: ["r_pan", "r_lift", "r_elbow", "r_wflex", "r_wroll", "r_grip"],
}

# Registers that legitimately differ servo to servo - not flagged.
PER_SERVO = {"ID", "Homing_Offset", "Present_Position", "Goal_Position", "Present_Load",
             "Present_Current", "Present_Temperature", "Present_Voltage", "Present_Velocity",
             "Moving", "Goal_Position_2", "Min_Position_Limit", "Max_Position_Limit"}

REGS = [
    # what the servo is doing now
    "Torque_Enable", "Status", "Moving", "Present_Position", "Goal_Position", "Present_Load",
    "Present_Current", "Present_Temperature", "Present_Voltage",
    # what limits it
    "Torque_Limit", "Max_Torque_Limit", "Min_Position_Limit", "Max_Position_Limit",
    "Protection_Current", "Protective_Torque", "Protection_Time", "Overload_Torque",
    "Over_Current_Protection_Time", "Max_Temperature_Limit", "Min_Voltage_Limit",
    "Max_Voltage_Limit", "Unloading_Condition",
    # how it moves
    "Operating_Mode", "Acceleration", "Goal_Velocity", "P_Coefficient", "D_Coefficient",
    "I_Coefficient", "Minimum_Startup_Force", "CW_Dead_Zone", "CCW_Dead_Zone",
    # identity
    "Homing_Offset", "Lock", "Baud_Rate", "Phase", "Angular_Resolution",
]

GRIPPER_SOFT = {"Torque_Limit", "Max_Torque_Limit", "Protection_Current", "Overload_Torque"}

STATUS_BITS = {0: "voltage", 1: "sensor", 2: "temperature", 3: "current", 4: "angle", 5: "overload"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", type=int, choices=(1, 2), default=2)
    args = ap.parse_args()

    names = NAMES[args.bus]
    motors = {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(names)}
    bus = FeetechMotorsBus(port=PORTS[args.bus], motors=motors)
    bus.connect(handshake=False)

    table: dict[str, dict[str, object]] = {}
    for reg in REGS:
        row = {}
        for n in names:
            try:
                row[n] = bus.read(reg, n, normalize=False, num_retry=3)
            except Exception as e:  # noqa: BLE001
                row[n] = "ERR"
        table[reg] = row
    bus.disconnect(disable_torque=False)

    w = 9
    print(f"\nbus {args.bus}  {PORTS[args.bus]}\n")
    print(f"{'register':<30}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (30 + w * len(names)))
    flags = []
    for reg in REGS:
        row = table[reg]
        vals = [row[n] for n in names]
        mark = ""
        if reg not in PER_SERVO:
            # Grippers are configured softer on purpose (lerobot lowers their
            # torque and protection limits so a closed gripper cannot cook
            # itself), so they are compared with nobody on those registers.
            peers = [n for n in names if not (n.endswith("grip") and reg in GRIPPER_SOFT)]
            common, count = Counter(row[n] for n in peers).most_common(1)[0]
            odd = [n for n in peers if row[n] != common]
            if odd and count > 1:
                mark = "  <- " + ", ".join(odd)
                for n in odd:
                    flags.append(f"{n}: {reg} = {row[n]} (others {common})")
        print(f"{reg:<30}" + "".join(f"{str(v):>{w}}" for v in vals) + mark)

    print()
    for n in names:
        st = table["Status"][n]
        if isinstance(st, int) and st:
            bits = [lbl for b, lbl in STATUS_BITS.items() if st & (1 << b)]
            flags.append(f"{n}: Status = {st} -> error bits: {', '.join(bits)}")
        lo, hi = table["Min_Position_Limit"][n], table["Max_Position_Limit"][n]
        if isinstance(lo, int) and isinstance(hi, int) and hi - lo < 500:
            flags.append(f"{n}: position limits {lo}..{hi} leave it almost no travel")
        tl = table["Torque_Limit"][n]
        if isinstance(tl, int) and tl < 100:
            flags.append(f"{n}: Torque_Limit = {tl} - it can barely push (1000 = full)")
    # Torque off everywhere is just "nothing has it enabled" - every lerobot
    # disconnect turns it off. Only an odd one out means anything.
    te = [table["Torque_Enable"][n] for n in names]
    if len(set(te)) > 1:
        for n in names:
            if table["Torque_Enable"][n] == 0:
                flags.append(f"{n}: torque is OFF while others are on")
    # Heat is the tell for a servo that is fighting something: a joint that
    # was told to move and could not draws stall current the whole time.
    temps = {n: table["Present_Temperature"][n] for n in names
             if isinstance(table["Present_Temperature"][n], int)}
    if len(temps) > 2:
        med = sorted(temps.values())[len(temps) // 2]
        for n, t in temps.items():
            if t - med >= 6:
                flags.append(f"{n}: {t} C, {t - med} C hotter than the others - it has been "
                             "pushing against something (stall, bind, or a stripped gear)")

    print("FINDINGS" if flags else "No servo differs from its neighbours.")
    for f in flags:
        print("  " + f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
