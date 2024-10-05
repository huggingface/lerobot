import threading
import time
from collections import deque
from threading import Thread

import numpy as np


class TemporalQueue:
    def __init__(self):
        self.items = deque(maxlen=10)
        self.timestamps = deque(maxlen=10)

    def add(self, item, timestamp):
        self.items.append(item)
        self.timestamps.append(timestamp)

    def get_latest(self):
        return self.items[-1], self.timestamps[-1]

    def get(self, timestamp):
        timestamps = np.array(list(self.timestamps))
        distances = np.abs(timestamps - timestamp)
        nearest_idx = distances.argmin()

        # print(float(distances[nearest_idx]))
        if float(distances[nearest_idx]) > 1 / 5:
            raise ValueError()

        return self.items[nearest_idx], self.timestamps[nearest_idx]

    def __len__(self):
        return len(self.items)


class Policy:
    def __init__(self):
        self.obs_queue = TemporalQueue()
        self.action_queue = TemporalQueue()
        self.thread = None

        self.n_action = 2
        FPS = 10  # noqa: N806
        self.delta_timestamps = [i / FPS for i in range(self.n_action)]

    def inference(self, observation):
        # TODO
        time.sleep(0.5)
        return [observation] * self.n_action

    def inference_loop(self):
        prev_timestamp = None
        while not self.stop_event.is_set():
            last_observation, last_timestamp = self.obs_queue.get_latest()

            if prev_timestamp is not None and prev_timestamp == last_timestamp:
                # in case inference ran faster than recording/adding a new observation in the queue
                time.sleep(0.1)
                continue

            pred_action_sequence = self.inference(last_observation)

            for action, delta_ts in zip(pred_action_sequence, self.delta_timestamps, strict=False):
                self.action_queue.add(action, last_timestamp + delta_ts)

            prev_timestamp = last_timestamp

    def select_action(
        self,
        new_observation: int,
    ) -> list[int]:
        present_time = time.time()
        self.obs_queue.add(new_observation, present_time)

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.inference_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        next_action = None
        while next_action is None:
            try:
                next_action = self.action_queue.get(present_time)
            except ValueError:
                time.sleep(0.1)  # no action available at this present time, we wait a bit

        return next_action


if __name__ == "__main__":
    time.sleep(1)
    policy = Policy()

    for new_observation in range(10):
        next_action = policy.select_action(new_observation)
        print(f"{new_observation=}, {next_action=}")
        time.sleep(0.5)  # frequency at which we receive a new observation (5 Hz = 0.2 s)
