# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import os

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.rerun_shutdown()


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


class _AutoScale:
    """Arrow scale that maps a slowly decaying peak to `max_length`, never below `margin` x the running RMS."""

    def __init__(self, max_length: float, peak_decay: float = 0.998, noise_alpha: float = 0.002, margin: float = 4.0):
        self.max_length = max_length
        self.peak_decay = peak_decay
        self.noise_alpha = noise_alpha
        self.margin = margin
        self.peak = 0.0
        self._noise_sq = 0.0

    def update(self, magnitude: float) -> tuple[float, float]:
        """Returns (scale in m per unit, level of `magnitude` in [0, 1])."""
        self._noise_sq += self.noise_alpha * (magnitude**2 - self._noise_sq)
        self.peak = max(magnitude, self.peak * self.peak_decay, self.margin * float(np.sqrt(self._noise_sq)))
        if self.peak <= 0:
            return 0.0, 0.0
        return self.max_length / self.peak, magnitude / self.peak


class TactileForce3DLogger:
    """
    Logs a live 3D view of the tactile finger to Rerun, all sensors being considered at the origin:
    - static: x/y/z frame, the rotation axis of each sensor (labelled) and the lever arm to the contact point
    - live: the rotation measured by each sensor (arrow along its axis), the resulting rotation vector, the force
      of each sensor (tip-to-tail, same colors as the time series) and the resulting force, their sum, applied at
      the end of the lever arm. The resulting force goes from grey to red with its magnitude.

    If the layout has a `timeseries_fn`, the 3 sensor signals are also plotted under `timeseries_entity` in a
    time series view showing a rolling window of the last `timeseries_duration_s` (sent as a blueprint, the other
    views keep being created automatically).

    Built from a robot's `tactile_layout` (see `SpectrobotTrifold.tactile_layout`). Arrows are scaled with
    `arrow_scale` (meters per unit) or, when it is None, each vector is auto-scaled on its recent peak.
    """

    _track_colors = ((255, 90, 90), (90, 220, 90), (90, 150, 255))  # same order as the RGB spectrogram

    def __init__(
        self,
        layout: dict,
        entity: str = "tactile_3d",
        timeseries_entity: str = "tactile_timeseries",
        max_arrow_length_m: float = 0.06,
    ):
        self.layout = layout
        self.entity = entity
        self.timeseries_entity = timeseries_entity
        self._rotation_scale = _AutoScale(max_arrow_length_m)
        self._force_scale = _AutoScale(max_arrow_length_m)
        self._static_logged = False
        self._timeseries_counts = None
        self._timeseries_paths = [
            f"{timeseries_entity}/{label.split()[0]}" for label in self.layout["sensor_labels"]
        ]

    @staticmethod
    def _heat(level: np.ndarray) -> np.ndarray:
        """Maps levels in [0, 1] to RGB colors, grey (idle) -> red (max)."""
        level = np.clip(level, 0.0, 1.0)[..., None]
        idle = np.array([150, 150, 150], dtype=np.float64)
        hot = np.array([255, 30, 30], dtype=np.float64)
        return (idle + (hot - idle) * level).astype(np.uint8)

    def _log_static(self, rr) -> None:
        origin = [(0.0, 0.0, 0.0)]
        axes = np.asarray(self.layout["sensor_axes"], dtype=np.float64)
        rr.log(self.entity, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            f"{self.entity}/axes",
            rr.Arrows3D(
                origins=origin * 3,
                vectors=[(0.08, 0.0, 0.0), (0.0, 0.08, 0.0), (0.0, 0.0, 0.08)],
                colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                radii=0.0005,
                labels=["x", "y", "z"],
            ),
            static=True,
        )
        rr.log(
            f"{self.entity}/sensor_axes",
            rr.Arrows3D(
                origins=origin * len(axes),
                vectors=-axes * 0.03,  # drawn on the negative side so they don't hide the live arrows
                colors=[(230, 200, 60)],
                radii=0.0008,
                labels=[f"{label} (rot)" for label in self.layout["sensor_labels"]],
            ),
            static=True,
        )
        rr.log(
            f"{self.entity}/lever_arm",
            rr.LineStrips3D([[origin[0], self.layout["lever_arm"]]], colors=[(90, 90, 90)], radii=0.002),
            static=True,
        )
        rr.log(f"{self.entity}/sensors", rr.Points3D(origin, radii=0.004, colors=[(230, 200, 60)]), static=True)

        if self.layout.get("timeseries_fn") is not None:
            import rerun.blueprint as rrb

            for path, color, label in zip(
                self._timeseries_paths, self._track_colors, self.layout["sensor_labels"], strict=False
            ):
                rr.log(path, rr.SeriesLines(colors=[color], names=[label]), static=True)
            window_s = float(self.layout.get("timeseries_duration_s", 2.0))
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.TimeSeriesView(
                        origin=self.timeseries_entity,
                        name=f"Tactile signals (last {window_s:g} s)",
                        time_ranges=rrb.VisibleTimeRange(
                            "log_time",
                            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-window_s),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        ),
                    ),
                    auto_views=True,
                    auto_layout=True,
                )
            )
        self._static_logged = True

    def __call__(self, observation: RobotObservation) -> None:
        require_package("rerun-sdk", extra="viz", import_name="rerun")
        import rerun as rr

        if not self._static_logged:
            self._log_static(rr)

        rotation = np.array([float(observation[k]) for k in self.layout["rotation_keys"]])
        force = np.array([float(observation[k]) for k in self.layout["force_keys"]])
        rotation_norm = float(np.linalg.norm(rotation))
        force_norm = float(np.linalg.norm(force))

        rotation_scale, rotation_level = self._rotation_scale.update(rotation_norm)
        force_scale, force_level = self._force_scale.update(force_norm)
        fixed_scale = self.layout.get("arrow_scale")
        if fixed_scale is not None:
            rotation_scale = force_scale = fixed_scale

        # Rotation measured by each sensor, drawn along its axis
        axes = np.asarray(self.layout["sensor_axes"], dtype=np.float64)
        signals = axes @ rotation  # back to per-sensor values (axes are orthonormal)
        component_levels = np.abs(signals) * rotation_scale / self._rotation_scale.max_length
        rr.log(
            f"{self.entity}/rotation_components",
            rr.Arrows3D(
                origins=[(0.0, 0.0, 0.0)] * len(axes),
                vectors=axes * signals[:, None] * rotation_scale,
                colors=self._heat(component_levels),
                radii=0.0015,
            ),
        )
        rr.log(
            f"{self.entity}/rotation",
            rr.Arrows3D(
                origins=[(0.0, 0.0, 0.0)],
                vectors=[rotation * rotation_scale],
                colors=[(80, 160, 255)],
                radii=0.0025,
                labels=[f"|rot|={rotation_norm:.3g}"],
            ),
        )
        # Force of each sensor, drawn tip-to-tail from the contact point so that the resulting force closes them
        sensor_force_matrix = self.layout.get("sensor_force_matrix")
        if sensor_force_matrix is not None:
            sensor_forces = (np.asarray(sensor_force_matrix, dtype=np.float64) * signals).T * force_scale
            origins = np.asarray(self.layout["lever_arm"], dtype=np.float64) + np.vstack(
                (np.zeros(3), np.cumsum(sensor_forces, axis=0)[:-1])
            )
            rr.log(
                f"{self.entity}/sensor_forces",
                rr.Arrows3D(
                    origins=origins,
                    vectors=sensor_forces,
                    colors=self._track_colors[: len(sensor_forces)],
                    radii=0.0015,
                    labels=[f"F {label.split()[0]}" for label in self.layout["sensor_labels"]],
                ),
            )
        rr.log(
            f"{self.entity}/force",
            rr.Arrows3D(
                origins=[self.layout["lever_arm"]],
                vectors=[force * force_scale],
                colors=self._heat(np.array([force_level])),
                radii=0.0025,
                labels=[f"|F|={force_norm:.3g}"],
            ),
        )
        rgb_key = self.layout.get("rgb_spectrogram_key")
        if rgb_key and rgb_key in observation:
            rr.log(f"{self.entity}/spectrogram_rgb", rr.Image(observation[rgb_key]))

        if self.layout.get("timeseries_fn") is not None:
            self._log_timeseries(rr)

    def _log_timeseries(self, rr) -> None:
        """Sends the new sensor samples as scalars on the `log_time` timeline (plotted in a rolling window)."""
        self._timeseries_counts, samples = self.layout["timeseries_fn"](self._timeseries_counts)
        for path, (times, values) in zip(self._timeseries_paths, samples, strict=True):
            if values.size:
                rr.send_columns(
                    path,
                    indexes=[rr.TimeColumn("log_time", timestamp=times)],
                    columns=rr.Scalars.columns(scalars=values),
                )
