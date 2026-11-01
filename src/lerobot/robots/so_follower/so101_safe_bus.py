"""SO101-specific authorization boundary for the serial motor bus."""

from contextlib import contextmanager
from os import environ

from lerobot.motors.motors_bus import SerialMotorsBus




# B43_7_9_PROXY
class _SO101RustyPotBusProxy:
    """Fail-closed proxy around the underlying RustyPot Bus."""

    READ_METHODS = frozenset({
        "scan",
        "broadcast_scan",
        "read_register",
        "read_register_with_error",
        "sync_read_register",
        "sync_read_register_with_error",
    })

    SAFE_METHODS = frozenset({
        "close",
    })

    SETUP_METHODS = frozenset({
        "change_id",
        "change_baudrate",
        "set_baudrate",
        "set_timeout",
    })

    ACTION_METHODS = frozenset({
        "write_register",
        "write_register_with_error",
        "sync_write_register",
    })

    __slots__ = ("__raw_bus", "__owner")

    _BLOCKED_INTERNALS = frozenset({
        "_raw",
        "_owner",
        "__raw_bus",
        "__owner",
        "_SO101RustyPotBusProxy__raw_bus",
        "_SO101RustyPotBusProxy__owner",
    })

    def __init__(self, raw_bus, owner):
        object.__setattr__(
            self, "_SO101RustyPotBusProxy__raw_bus", raw_bus
        )
        object.__setattr__(
            self, "_SO101RustyPotBusProxy__owner", owner
        )

    def __getattribute__(self, name):
        if name in object.__getattribute__(
            self, "_BLOCKED_INTERNALS"
        ):
            raise AttributeError(
                f"SO101 proxy: internal attribute hidden: {name}"
            )
        return object.__getattribute__(self, name)

    @staticmethod
    def _argument(args, kwargs, index, name):
        if len(args) > index:
            return args[index]
        return kwargs.get(name)

    def __getattr__(self, name):
        allowed = (
            self.READ_METHODS
            | self.SAFE_METHODS
            | self.SETUP_METHODS
            | self.ACTION_METHODS
            | {"set_torque"}
        )

        if name not in allowed:
            raise RuntimeError(
                f"SO101 proxy: underlying Bus method rejected: {name}"
            )

        raw_bus = object.__getattribute__(
            self, "_SO101RustyPotBusProxy__raw_bus"
        )
        owner = object.__getattribute__(
            self, "_SO101RustyPotBusProxy__owner"
        )

        method = getattr(raw_bus, name)

        if not callable(method):
            raise RuntimeError(
                f"SO101 proxy: non-callable interface rejected: {name}"
            )

        def guarded(*args, **kwargs):

            if name in self.READ_METHODS or name in self.SAFE_METHODS:
                pass

            elif name == "set_torque":
                enabled = self._argument(
                    args, kwargs, 1, "enabled"
                )

                if enabled is True:
                    owner._require(
                        owner.TORQUE_ENV,
                        "RustyPot torque enable"
                    )
                elif enabled is not False:
                    raise RuntimeError(
                        "SO101 proxy: invalid torque state"
                    )

            elif name in self.ACTION_METHODS:
                register = self._argument(
                    args, kwargs, 1, "name"
                )
                owner._authorize_register_write(
                    register,
                    f"RustyPot {name}"
                )

            elif name in self.SETUP_METHODS:
                owner._require(
                    owner.SETUP_ENV,
                    f"RustyPot {name}"
                )

            else:
                raise RuntimeError(
                    f"SO101 proxy: operation rejected: {name}"
                )

            return method(*args, **kwargs)

        return guarded

    def __setattr__(self, name, value):
        raise RuntimeError(
            f"SO101 proxy: direct attribute modification rejected: {name}"
        )


class SO101SafeSerialMotorsBus(SerialMotorsBus):
    """Restrict SO101 connection, setup and torque operations independently."""

    CONNECT_ENV = "SO101_HARDWARE_CONNECT_APPROVED"
    SETUP_ENV = "SO101_HARDWARE_SETUP_APPROVED"
    TORQUE_ENV = "SO101_HARDWARE_TORQUE_APPROVED"
    ACTION_ENV = "SO101_HARDWARE_APPROVED"

    @staticmethod
    def _require(env_name: str, operation: str) -> None:
        if environ.get(env_name) != "1":
            raise RuntimeError(
                f"SO101 safety gate: {operation} disabled; "
                f"set {env_name}=1 only after explicit safety authorization."
            )

    def _open_bus(self):
        """Prevent direct access to the underlying serial bus."""
        self._require(self.CONNECT_ENV, "low-level bus opening")
        return super()._open_bus()

    ACTION_REGISTERS = frozenset({
        "goal_position",
        "goal_velocity",
        "goal_current",
        "goal_torque",
        "goal_pwm",
    })

    CONFIG_REGISTERS = frozenset({
        "operating_mode",
        "p_coefficient",
        "i_coefficient",
        "d_coefficient",
        "max_torque_limit",
        "protection_current",
        "overload_torque",
        "drive_mode",
        "homing_offset",
        "min_position_limit",
        "max_position_limit",
        "return_delay_time",
        "phase",
        "acceleration",
        "maximum_acceleration",
    })

    def _authorize_register_write(self, data_name, operation):
        """Fail closed unless a register belongs to an explicit allowlist."""
        if not isinstance(data_name, str):
            raise RuntimeError(
                "SO101 safety gate: missing or invalid register name."
            )

        register = data_name.strip().lower()

        if register in self.ACTION_REGISTERS:
            self._require(self.ACTION_ENV, operation)
            return

        if register in self.CONFIG_REGISTERS:
            self._require(self.SETUP_ENV, "motor register write")
            return

        raise RuntimeError(
            f"SO101 safety gate: unclassified register rejected: {data_name}"
        )

    def sync_write(self, *args, **kwargs):
        """Authorize synchronous writes according to the target register."""
        data_name = args[0] if args else kwargs.get("data_name")
        self._authorize_register_write(data_name, "motor action dispatch")
        return super().sync_write(*args, **kwargs)

    def _connect(self, *args, **kwargs):
        """Protect connection and wrap the connected RustyPot bus."""
        self._require(self.CONNECT_ENV, "low-level serial connection")

        result = super()._connect(*args, **kwargs)

        if self._bus is not None:
            if not isinstance(self._bus, _SO101RustyPotBusProxy):
                self._bus = _SO101RustyPotBusProxy(
                    self._bus, self
                )

        return result

    def connect(self, *args, **kwargs):
        self._require(self.CONNECT_ENV, "hardware connection")
        # connect() also runs calibration/configuration in SOFollower.
        self._require(self.SETUP_ENV, "motor setup")
        return super().connect(*args, **kwargs)

    def _set_torque(self, motors, enabled, num_retry=0):
        """Enforce torque authorization at the lowest shared torque entry."""
        if enabled is True:
            self._require(self.TORQUE_ENV, "low-level torque enable")
        elif enabled is not False:
            raise RuntimeError(
                "SO101 safety gate: torque state must be an explicit boolean."
            )

        return super()._set_torque(motors, enabled, num_retry)

    def enable_torque(self, *args, **kwargs):
        self._require(self.TORQUE_ENV, "torque enable")
        return super().enable_torque(*args, **kwargs)

    def setup_motor(self, *args, **kwargs):
        self._require(self.SETUP_ENV, "single motor setup")
        return super().setup_motor(*args, **kwargs)

    def setup_motors(self, *args, **kwargs):
        self._require(self.SETUP_ENV, "motor setup")
        return super().setup_motors(*args, **kwargs)

    def record_calibration(self, *args, **kwargs):
        self._require(self.SETUP_ENV, "motor calibration")
        return super().record_calibration(*args, **kwargs)

    def write_calibration(self, *args, **kwargs):
        self._require(self.SETUP_ENV, "calibration write")
        return super().write_calibration(*args, **kwargs)

    def configure_motors(self, *args, **kwargs):
        self._require(self.SETUP_ENV, "motor configuration")
        return super().configure_motors(*args, **kwargs)

    def set_operating_mode(self, *args, **kwargs):
        self._require(self.SETUP_ENV, "operating mode change")
        return super().set_operating_mode(*args, **kwargs)

    def write(self, *args, **kwargs):
        """Authorize individual writes according to the target register."""
        data_name = args[0] if args else kwargs.get("data_name")
        self._authorize_register_write(data_name, "motor action dispatch")
        return super().write(*args, **kwargs)

    def set_baudrate(self, *args, **kwargs):
        self._require(self.SETUP_ENV, "motor baudrate change")
        return super().set_baudrate(*args, **kwargs)

    @contextmanager
    def torque_disabled(self, motors=None):
        """Keep torque disabled after the block, including exceptional exit."""
        self.disable_torque(motors)
        yield
        # Deliberately do not restore torque automatically.
