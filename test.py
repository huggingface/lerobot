import math
import threading
import time
from threading import Thread


class TemporalQueue:
    def __init__(self):
        self.items = []
        self.timestamps = []

    def add(self, item, timestamp):
        self.items.append(item)
        self.timestamps.append(timestamp)

    def get(self, timestamp=None):
        if timestamp is None:
            return self.items[-1], self.timestamps[-1]

        # TODO(rcadene): implement nearest neighbor instead of hacky floor
        for idx, t in list(enumerate(self.timestamps))[::-1]:
            if math.floor(t) == math.floor(timestamp):
                return self.items[idx], t

        raise ValueError()

    def __len__(self):
        return len(self.items)


class Policy:
    def __init__(self):
        self.obs_queue = TemporalQueue()
        self.action_queue = TemporalQueue()
        self.thread = None

    def inference(self, observation):
        # TODO
        time.sleep(0.5)
        return observation

    def inference_loop(self):
        previous_timestamp = None
        while not self.stop_event.is_set():
            latest_observation, latest_timestamp = self.obs_queue.get()

            if previous_timestamp is not None and previous_timestamp == latest_timestamp:
                time.sleep(
                    0.1
                )  # in case inference ran faster than recording/adding a new observation in the queue
            else:
                predicted_action_sequence = self.inference(latest_observation)
                self.action_queue.add(predicted_action_sequence, latest_timestamp)
                previous_timestamp = latest_timestamp

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
