import torch
from datetime import datetime
from typing import Any

import rerun as rr
import rerun.blueprint as rrb


class RerunLogger:
    """
    A fully automatic Rerun logger designed to parse and visualize step
    dictionaries directly from a LeRobotDataset.
    """

    def __init__(
        self,
        prefix: str = "",
        memory_limit: str = "200MB",
        idxrangeboundary: int | None = 300,
    ):
        """Initializes the Rerun logger."""
        # Use a descriptive name for the Rerun recording
        rr.init(f"Dataset_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        rr.spawn(memory_limit=memory_limit)

        self.prefix = prefix
        self.blueprint_sent = False
        self.idxrangeboundary = idxrangeboundary

        # --- Internal cache for discovered keys ---
        self._image_keys: tuple[str, ...] = ()
        self._state_key: str = ""
        self._action_key: str = ""
        self._index_key: str = "index"
        self._task_key: str = "task"
        self._episode_index_key: str = "episode_index"

        self.current_episode = -1

    def _initialize_from_data(self, step_data: dict[str, Any]):
        """Inspects the first data dictionary to discover components and set up the blueprint."""
        print("RerunLogger: First data packet received. Auto-configuring...")

        image_keys = []
        for key, value in step_data.items():
            if key.startswith("observation.images.") and isinstance(value, torch.Tensor) and value.ndim > 2:
                image_keys.append(key)
            elif key == "observation.state":
                self._state_key = key
            elif key == "action":
                self._action_key = key

        self._image_keys = tuple(sorted(image_keys))

        if "index" in step_data:
            self._index_key = "index"
        elif "frame_index" in step_data:
            self._index_key = "frame_index"

        print(f"  - Using '{self._index_key}' for time sequence.")
        print(f"  - Detected State Key: '{self._state_key}'")
        print(f"  - Detected Action Key: '{self._action_key}'")
        print(f"  - Detected Image Keys: {self._image_keys}")
        if self.idxrangeboundary:
            self.setup_blueprint()

    def setup_blueprint(self):
        """Sets up and sends the Rerun blueprint based on detected components."""
        views = []

        for key in self._image_keys:
            clean_name = key.replace("observation.images.", "")
            entity_path = f"{self.prefix}images/{clean_name}"
            views.append(rrb.Spatial2DView(origin=entity_path, name=clean_name))

        if self._state_key:
            entity_path = f"{self.prefix}state"
            views.append(
                rrb.TimeSeriesView(
                    origin=entity_path,
                    name="Observation State",
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "frame",
                            start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.idxrangeboundary),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                    plot_legend=rrb.PlotLegend(visible=True),
                )
            )

        if self._action_key:
            entity_path = f"{self.prefix}action"
            views.append(
                rrb.TimeSeriesView(
                    origin=entity_path,
                    name="Action",
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "frame",
                            start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.idxrangeboundary),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                    plot_legend=rrb.PlotLegend(visible=True),
                )
            )

        if not views:
            print("Warning: No visualizable components detected in the data.")
            return

        grid = rrb.Grid(contents=views)
        rr.send_blueprint(grid)
        self.blueprint_sent = True

    def log_step(self, step_data: dict[str, Any]):
        """Logs a single step dictionary from your dataset."""
        if not self.blueprint_sent:
            self._initialize_from_data(step_data)

        if self._index_key in step_data:
            current_index = step_data[self._index_key].item()
            rr.set_time_sequence("frame", current_index)

        episode_idx = step_data.get(self._episode_index_key, torch.tensor(-1)).item()
        if episode_idx != self.current_episode:
            self.current_episode = episode_idx
            task_name = step_data.get(self._task_key, "Unknown Task")
            log_text = f"Starting Episode {self.current_episode}: {task_name}"
            rr.log(f"{self.prefix}info/task", rr.TextLog(log_text, level=rr.TextLogLevel.INFO))

        for key in self._image_keys:
            if key in step_data:
                image_tensor = step_data[key]
                if image_tensor.ndim > 2:
                    clean_name = key.replace("observation.images.", "")
                    entity_path = f"{self.prefix}images/{clean_name}"
                    if image_tensor.shape[0] in [1, 3, 4]:
                        image_tensor = image_tensor.permute(1, 2, 0)
                    rr.log(entity_path, rr.Image(image_tensor))

        if self._state_key in step_data:
            state_tensor = step_data[self._state_key]
            entity_path = f"{self.prefix}state"
            for i, val in enumerate(state_tensor):
                rr.log(f"{entity_path}/joint_{i}", rr.Scalar(val.item()))

        if self._action_key in step_data:
            action_tensor = step_data[self._action_key]
            entity_path = f"{self.prefix}action"
            for i, val in enumerate(action_tensor):
                rr.log(f"{entity_path}/joint_{i}", rr.Scalar(val.item()))


def visualization_data(idx, observation, state, action, online_logger):
    item_data: dict[str, Any] = {
        "index": torch.tensor(idx),
        "observation.state": state,
        "action": action,
    }
    for k, v in observation.items():
        if k not in ("index", "observation.state", "action"):
            item_data[k] = v
    online_logger.log_step(item_data)
