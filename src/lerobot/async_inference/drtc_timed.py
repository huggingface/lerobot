from dataclasses import dataclass

from .helpers import Action, RawObservation


@dataclass
class DrtcTimedData:
    """Base timing payload for the DRTC control-step clock."""

    timestamp: float
    control_step: int

    def get_timestamp(self):
        return self.timestamp

    def get_control_step(self):
        return self.control_step


@dataclass
class DrtcAction(DrtcTimedData):
    """A DRTC action identified by its source control step and execution step."""

    action_step: int
    action: Action = None

    def get_action_step(self):
        return self.action_step

    def get_action(self):
        return self.action


@dataclass
class DrtcObservation(DrtcTimedData):
    """A DRTC observation that carries the target chunk start step."""

    observation: RawObservation = None
    chunk_start_step: int = 0
    server_received_ts: float = 0.0

    def get_observation(self):
        return self.observation
