import math
from dataclasses import dataclass


class GripperSafetyError(RuntimeError):
    pass


@dataclass
class GripperStatus:
    load: float
    current: float
    temperature: float
    voltage: float
    overload_detected: bool = False


class GripperSafetyGuard:
    def __init__(self):
        self.fault_latched = False
        self.fault_reason = None

    def _fault(self, reason):
        self.fault_latched = True
        self.fault_reason = reason
        raise GripperSafetyError(reason)

    def check(self, status):
        if self.fault_latched:
            raise GripperSafetyError(
                f"GRIPPER LOCKED: {self.fault_reason}"
            )

        if status is None:
            self._fault("Gripper status unavailable")

        for name in (
            "load",
            "current",
            "temperature",
            "voltage",
        ):
            try:
                value = float(getattr(status, name))
            except (TypeError, ValueError, OverflowError, AttributeError):
                self._fault(f"Invalid gripper status: {name}")

            if not math.isfinite(value):
                self._fault(f"Non-finite gripper status: {name}")

        if type(status.overload_detected) is not bool:
            self._fault("Invalid overload flag")

        if status.overload_detected:
            self._fault("Gripper overload detected")

        return True


def run_test(name, statuses, expected_fault):
    guard = GripperSafetyGuard()
    fault = False

    try:
        for status in statuses:
            guard.check(status)
    except GripperSafetyError as exc:
        fault = True
        print(name, "FAULT:", exc)

    assert fault == expected_fault, name
    assert guard.fault_latched == expected_fault, name

    if expected_fault:
        try:
            guard.check(
                GripperStatus(0, 0, 25, 12)
            )
        except GripperSafetyError:
            print(name, "LATCH CONFIRMED")
        else:
            raise AssertionError(
                f"{name}: fault latch failed"
            )

    print(name, "PASS")


if __name__ == "__main__":
    print("===== SAFETY-076 GRIPPER GUARD MOCK TEST =====")

    normal = GripperStatus(
        load=100,
        current=100,
        temperature=30,
        voltage=12,
    )

    run_test(
        "NORMAL",
        [normal],
        False,
    )

    run_test(
        "OVERLOAD",
        [
            normal,
            GripperStatus(
                100, 100, 30, 12,
                overload_detected=True,
            ),
        ],
        True,
    )

    run_test(
        "STATUS_MISSING",
        [None],
        True,
    )

    run_test(
        "LOAD_NAN",
        [GripperStatus(float("nan"), 100, 30, 12)],
        True,
    )

    run_test(
        "CURRENT_INF",
        [GripperStatus(100, float("inf"), 30, 12)],
        True,
    )

    run_test(
        "TEMPERATURE_NAN",
        [GripperStatus(100, 100, float("nan"), 12)],
        True,
    )

    run_test(
        "INVALID_OVERLOAD_FLAG",
        [GripperStatus(100, 100, 30, 12, "false")],
        True,
    )

    print()
    print("===== SAFETY-076 ALL MOCK TESTS PASSED =====")
    print("REAL ROBOT CONNECTED: NO")
    print("REAL ROBOT COMMANDS SENT: NONE")
