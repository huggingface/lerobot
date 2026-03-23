"""
Waypoint-following controller for the LeKiwi three-wheel omnidirectional base.

This module provides a standalone waypoint controller that converts a sequence of
(x, y, theta) waypoints into body-frame velocity commands compatible with LeKiwi's
`send_action()` interface.

Usage (standalone simulation):
    python waypoint_controller.py

Usage (with real robot):
    from lerobot.robots.lekiwi.waypoint_controller import WaypointController
    controller = WaypointController()
    controller.set_waypoints([(1.0, 0.0, 0.0), (1.0, 1.0, 90.0)])
    # In your control loop:
    vel_cmd = controller.update(current_x, current_y, current_theta, dt)
    action = {**arm_positions, **vel_cmd}
    robot.send_action(action)
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Kiwi drive kinematics (mirrors lekiwi.py exactly)
# ---------------------------------------------------------------------------

WHEEL_ANGLES_DEG = np.array([240.0, 0.0, 120.0])
WHEEL_ANGLES_RAD = np.radians(WHEEL_ANGLES_DEG) - np.pi / 2
DEFAULT_WHEEL_RADIUS = 0.05  # meters
DEFAULT_BASE_RADIUS = 0.125  # meters
STEPS_PER_DEG = 4096.0 / 360.0
MAX_RAW = 3000


def build_kinematic_matrix(base_radius: float = DEFAULT_BASE_RADIUS) -> np.ndarray:
    """3x3 matrix that maps [vx, vy, omega_rad] -> wheel linear speeds."""
    return np.array([[math.cos(a), math.sin(a), base_radius] for a in WHEEL_ANGLES_RAD])


def body_vel_to_wheel_raw(
    vx: float,
    vy: float,
    omega_degps: float,
    wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    base_radius: float = DEFAULT_BASE_RADIUS,
    max_raw: int = MAX_RAW,
) -> dict[str, int]:
    """Convert body-frame velocity to raw wheel motor commands.

    Args:
        vx: Forward velocity (m/s).
        vy: Lateral velocity (m/s, +left).
        omega_degps: Rotational velocity (deg/s, +CCW).
        wheel_radius: Wheel radius in meters.
        base_radius: Distance from center to each wheel in meters.
        max_raw: Maximum raw motor ticks before proportional scaling.

    Returns:
        Dict with keys base_left_wheel, base_back_wheel, base_right_wheel.
    """
    omega_rad = math.radians(omega_degps)
    vel = np.array([vx, vy, omega_rad])

    m = build_kinematic_matrix(base_radius)
    wheel_linear = m @ vel
    wheel_degps = np.degrees(wheel_linear / wheel_radius)

    # Proportional scaling if any wheel exceeds max_raw
    raw_floats = np.abs(wheel_degps) * STEPS_PER_DEG
    max_computed = raw_floats.max()
    if max_computed > max_raw:
        wheel_degps *= max_raw / max_computed

    def degps_to_raw(d: float) -> int:
        v = int(round(d * STEPS_PER_DEG))
        return max(-0x8000, min(0x7FFF, v))

    raw = [degps_to_raw(d) for d in wheel_degps]
    return {
        "base_left_wheel": raw[0],
        "base_back_wheel": raw[1],
        "base_right_wheel": raw[2],
    }


def wheel_raw_to_body_vel(
    left: int,
    back: int,
    right: int,
    wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    base_radius: float = DEFAULT_BASE_RADIUS,
) -> dict[str, float]:
    """Convert raw wheel feedback to body-frame velocity (inverse of above)."""

    # raw_to_degps = lambda r: r / STEPS_PER_DEG
    def raw_to_degps(r):
        return r / STEPS_PER_DEG

    wheel_degps = np.array([raw_to_degps(left), raw_to_degps(back), raw_to_degps(right)])
    wheel_radps = np.radians(wheel_degps)
    wheel_linear = wheel_radps * wheel_radius

    m = build_kinematic_matrix(base_radius)
    body = np.linalg.inv(m) @ wheel_linear
    return {"x.vel": body[0], "y.vel": body[1], "theta.vel": math.degrees(body[2])}


# ---------------------------------------------------------------------------
# Odometry tracker (dead-reckoning from wheel feedback)
# ---------------------------------------------------------------------------


@dataclass
class Odometry:
    """Simple 2D dead-reckoning odometry from body-frame velocities."""

    x: float = 0.0  # meters, world frame
    y: float = 0.0  # meters, world frame
    theta: float = 0.0  # degrees, world frame (+CCW)

    def update(self, vx_body: float, vy_body: float, omega_degps: float, dt: float) -> None:
        """Integrate body-frame velocities over one timestep.

        Args:
            vx_body: Forward velocity in body frame (m/s).
            vy_body: Lateral velocity in body frame (m/s).
            omega_degps: Rotational velocity (deg/s).
            dt: Timestep in seconds.
        """
        theta_rad = math.radians(self.theta)
        # Rotate body velocity to world frame
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        vx_world = cos_t * vx_body - sin_t * vy_body
        vy_world = sin_t * vx_body + cos_t * vy_body

        self.x += vx_world * dt
        self.y += vy_world * dt
        self.theta += omega_degps * dt
        # Normalize to [-180, 180)
        self.theta = (self.theta + 180.0) % 360.0 - 180.0

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.theta = theta


# ---------------------------------------------------------------------------
# PID controller (single-axis)
# ---------------------------------------------------------------------------


@dataclass
class PIDController:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    max_output: float = float("inf")
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float = field(default=0.0, init=False, repr=False)

    def compute(self, error: float, dt: float) -> float:
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.max_output, min(self.max_output, output))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


# ---------------------------------------------------------------------------
# Waypoint controller
# ---------------------------------------------------------------------------


@dataclass
class WaypointControllerConfig:
    # PID gains for linear (xy) control
    kp_linear: float = 1.5
    ki_linear: float = 0.0
    kd_linear: float = 0.3

    # PID gains for angular (theta) control
    kp_angular: float = 2.0
    ki_angular: float = 0.0
    kd_angular: float = 0.2

    # Velocity limits
    max_linear_vel: float = 0.3  # m/s
    max_angular_vel: float = 90.0  # deg/s

    # Waypoint reached thresholds
    position_tolerance: float = 0.02  # meters
    angle_tolerance: float = 3.0  # degrees

    # When closer than this, start decelerating (smooth approach)
    decel_radius: float = 0.15  # meters


class WaypointController:
    """Follows a list of (x, y, theta) waypoints using PID control.

    The controller outputs body-frame velocity commands (x.vel, y.vel, theta.vel)
    that can be fed directly into LeKiwi's send_action().

    Typical integration:

        controller = WaypointController()
        controller.set_waypoints([(1.0, 0.0, 0.0), (1.0, 1.0, 90.0)])
        odom = Odometry()

        while controller.has_waypoints():
            obs = robot.get_observation()
            odom.update(obs["x.vel"], obs["y.vel"], obs["theta.vel"], dt)

            vel_cmd = controller.update(odom.x, odom.y, odom.theta, dt)
            action = {
                "arm_shoulder_pan.pos": obs["arm_shoulder_pan.pos"],
                "arm_shoulder_lift.pos": obs["arm_shoulder_lift.pos"],
                "arm_elbow_flex.pos": obs["arm_elbow_flex.pos"],
                "arm_wrist_flex.pos": obs["arm_wrist_flex.pos"],
                "arm_wrist_roll.pos": obs["arm_wrist_roll.pos"],
                "arm_gripper.pos": obs["arm_gripper.pos"],
                **vel_cmd,
            }
            robot.send_action(action)
    """

    def __init__(self, config: WaypointControllerConfig | None = None):
        self.config = config or WaypointControllerConfig()
        self._waypoints: list[tuple[float, float, float]] = []
        self._current_idx: int = 0

        self._pid_x = PIDController(
            kp=self.config.kp_linear,
            ki=self.config.ki_linear,
            kd=self.config.kd_linear,
            max_output=self.config.max_linear_vel,
        )
        self._pid_y = PIDController(
            kp=self.config.kp_linear,
            ki=self.config.ki_linear,
            kd=self.config.kd_linear,
            max_output=self.config.max_linear_vel,
        )
        self._pid_theta = PIDController(
            kp=self.config.kp_angular,
            ki=self.config.ki_angular,
            kd=self.config.kd_angular,
            max_output=self.config.max_angular_vel,
        )

    def set_waypoints(self, waypoints: list[tuple[float, float, float]]) -> None:
        """Set the waypoint list. Each waypoint is (x_meters, y_meters, theta_degrees)."""
        self._waypoints = list(waypoints)
        self._current_idx = 0
        self._pid_x.reset()
        self._pid_y.reset()
        self._pid_theta.reset()

    def has_waypoints(self) -> bool:
        return self._current_idx < len(self._waypoints)

    @property
    def current_waypoint(self) -> tuple[float, float, float] | None:
        if self.has_waypoints():
            return self._waypoints[self._current_idx]
        return None

    @property
    def progress(self) -> str:
        total = len(self._waypoints)
        idx = min(self._current_idx, total)
        return f"{idx}/{total}"

    @staticmethod
    def _angle_diff(target_deg: float, current_deg: float) -> float:
        """Shortest signed angle difference in degrees, range (-180, 180]."""
        diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
        return diff

    def update(self, x: float, y: float, theta: float, dt: float) -> dict[str, float]:
        """Compute body-frame velocity command to reach the current waypoint.

        Args:
            x: Current x position in world frame (meters).
            y: Current y position in world frame (meters).
            theta: Current heading in world frame (degrees, +CCW).
            dt: Timestep since last call (seconds).

        Returns:
            Dict with keys x.vel (m/s), y.vel (m/s), theta.vel (deg/s).
            Returns zero velocities if no waypoints remain.
        """
        stop = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

        if not self.has_waypoints() or dt <= 0:
            return stop

        wx, wy, wtheta = self._waypoints[self._current_idx]

        # --- Position error in world frame ---
        dx_world = wx - x
        dy_world = wy - y
        dist = math.hypot(dx_world, dy_world)
        angle_err = self._angle_diff(wtheta, theta)

        # --- Check if waypoint reached ---
        if dist < self.config.position_tolerance and abs(angle_err) < self.config.angle_tolerance:
            self._current_idx += 1
            self._pid_x.reset()
            self._pid_y.reset()
            self._pid_theta.reset()
            if not self.has_waypoints():
                return stop
            # Recurse with the next waypoint
            return self.update(x, y, theta, dt)

        # --- Transform world-frame error to body frame ---
        theta_rad = math.radians(theta)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        dx_body = cos_t * dx_world + sin_t * dy_world
        dy_body = -sin_t * dx_world + cos_t * dy_world

        # --- Smooth deceleration near waypoint ---
        decel_scale = min(1.0, dist / self.config.decel_radius) if self.config.decel_radius > 0 else 1.0

        # --- PID on body-frame errors ---
        vx_body = self._pid_x.compute(dx_body, dt) * decel_scale
        vy_body = self._pid_y.compute(dy_body, dt) * decel_scale
        omega = self._pid_theta.compute(angle_err, dt)

        # --- Clamp velocities ---
        linear_speed = math.hypot(vx_body, vy_body)
        if linear_speed > self.config.max_linear_vel:
            scale = self.config.max_linear_vel / linear_speed
            vx_body *= scale
            vy_body *= scale

        return {"x.vel": vx_body, "y.vel": vy_body, "theta.vel": omega}


# ---------------------------------------------------------------------------
# Convenience: run a full waypoint sequence on a connected LeKiwi robot
# ---------------------------------------------------------------------------


def follow_waypoints(
    robot,
    waypoints: list[tuple[float, float, float]],
    config: WaypointControllerConfig | None = None,
    loop_freq_hz: float = 30.0,
    verbose: bool = True,
) -> Odometry:
    """Drive a connected LeKiwi robot through a waypoint sequence.

    Args:
        robot: A connected LeKiwi or LeKiwiClient instance.
        waypoints: List of (x, y, theta) targets.
        config: Optional controller tuning.
        loop_freq_hz: Control loop frequency.
        verbose: Print progress to stdout.

    Returns:
        Final odometry state.
    """
    controller = WaypointController(config)
    controller.set_waypoints(waypoints)
    odom = Odometry()

    dt = 1.0 / loop_freq_hz
    arm_keys = [
        "arm_shoulder_pan.pos",
        "arm_shoulder_lift.pos",
        "arm_elbow_flex.pos",
        "arm_wrist_flex.pos",
        "arm_wrist_roll.pos",
        "arm_gripper.pos",
    ]

    try:
        while controller.has_waypoints():
            loop_start = time.perf_counter()

            # Read current state
            obs = robot.get_observation()

            # Update odometry from wheel feedback
            odom.update(obs["x.vel"], obs["y.vel"], obs["theta.vel"], dt)

            # Compute velocity command
            vel_cmd = controller.update(odom.x, odom.y, odom.theta, dt)

            # Build full action (hold arm in place + base velocity)
            action = {k: obs.get(k, 0.0) for k in arm_keys}
            action.update(vel_cmd)

            robot.send_action(action)

            if verbose:
                wp = controller.current_waypoint
                wp_str = f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.1f}°)" if wp else "DONE"
                print(
                    f"[{controller.progress}] "
                    f"pos=({odom.x:.3f}, {odom.y:.3f}, {odom.theta:.1f}°) "
                    f"vel=({vel_cmd['x.vel']:.3f}, {vel_cmd['y.vel']:.3f}, {vel_cmd['theta.vel']:.1f}°/s) "
                    f"target={wp_str}",
                    end="\r",
                )

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user.")
    finally:
        # Stop the base
        robot.send_action(
            {
                **dict.fromkeys(arm_keys, 0.0),
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }
        )

    if verbose:
        print(f"\nFinal position: ({odom.x:.3f}, {odom.y:.3f}, {odom.theta:.1f}°)")

    return odom


# ---------------------------------------------------------------------------
# Standalone simulation demo
# ---------------------------------------------------------------------------


def _simulate():
    """Run a simple simulation to visualize the waypoint controller."""
    print("=== LeKiwi Waypoint Controller — Simulation ===\n")

    waypoints = [
        (0.5, 0.0, 0.0),  # drive 0.5m forward
        (0.5, 0.5, 90.0),  # strafe left 0.5m and turn 90°
        (0.0, 0.5, 180.0),  # drive back
        (0.0, 0.0, 0.0),  # return to origin
    ]

    controller = WaypointController()
    controller.set_waypoints(waypoints)
    odom = Odometry()

    dt = 1.0 / 30.0  # 30 Hz
    step = 0
    max_steps = 30 * 60  # 60 seconds max

    print(f"Waypoints: {waypoints}\n")
    print(f"{'Step':>5}  {'Pos X':>7}  {'Pos Y':>7}  {'Theta':>7}  {'Vx':>7}  {'Vy':>7}  {'Omega':>7}  WP")
    print("-" * 75)

    while controller.has_waypoints() and step < max_steps:
        vel = controller.update(odom.x, odom.y, odom.theta, dt)

        # Simulate: apply velocity to odometry (perfect actuation)
        odom.update(vel["x.vel"], vel["y.vel"], vel["theta.vel"], dt)

        if step % 30 == 0:  # Print every second
            wp = controller.current_waypoint
            wp_str = f"({wp[0]:.1f},{wp[1]:.1f},{wp[2]:.0f}°)" if wp else "DONE"
            print(
                f"{step:5d}  {odom.x:7.3f}  {odom.y:7.3f}  {odom.theta:7.1f}  "
                f"{vel['x.vel']:7.3f}  {vel['y.vel']:7.3f}  {vel['theta.vel']:7.1f}  {wp_str}"
            )

        # Also compute raw motor commands (for verification)
        raw = body_vel_to_wheel_raw(vel["x.vel"], vel["y.vel"], vel["theta.vel"])
        step += 1

    print("-" * 75)
    print(f"Completed in {step} steps ({step * dt:.1f}s)")
    print(f"Final position: ({odom.x:.4f}, {odom.y:.4f}, {odom.theta:.2f}°)")
    print(f"Last raw motor cmds: {raw}")


if __name__ == "__main__":
    _simulate()
