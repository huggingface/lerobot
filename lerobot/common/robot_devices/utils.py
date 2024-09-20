import time


def precise_sleep(seconds: float, blocking: bool = False):
    """A more precise sleep than time.sleep

    There are several factors that influence the precision of time.sleep. They basically boil down to the OS
    timer granularity (see https://docs.python.org/3.11/whatsnew/3.11.html#time), and the system load which
    influences CPU scheduling.

    In all cases, nothing much can be done about issues with CPU scheduling. To be as precise as possible,
    we need to use a while-loop with a `pass`. This unfortunately keeps the CPU busy the whole time.

    The trick we use here, is to sleep most of the requested time, but switch to blocking the CPU towards the
    end of the requested time.

    NOTE: See the benchmarking script at the bottom of the source file.

    Args:
        seconds: Amount of time to sleep for in seconds.
        blocking: If set to True, just does a while-loop with `pass` making the sleep as precise as possible
            but using the CPU the whole time. Set this to True, if you don't need to parallelize with other
            work.
    """
    end_time = time.perf_counter() + seconds
    # 20 ms buffer is usually enough to account for an overly-long last sleep, even with heavy system load.
    end_time_with_buffer = end_time - 0.02
    while (now := time.perf_counter()) < end_time:
        if now < end_time_with_buffer and not blocking:
            # 100 microsec is faster than any control loop frequency we expect, but long enough to not hog
            # the CPU.
            time.sleep(0.0001)
        else:
            pass


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    """
    Simply run this benchmark with:

    ```bash
    python lerobot/common/robot_devices/utils.py
    ```

    OR, to test it in the presence of high system load, set N_CORES to the number of cores you want running
    at 100%:

    ```bash
    N_CORES=16; for i in $(seq 1 $N_CORES); do yes > /dev/null & done
    python lerobot/common/robot_devices/utils.py
    killall yes
    ```

    `precise_sleep` should be better than `sleep` even up to maximal core usage.
    """

    import math

    from tqdm import trange

    sleep_time = 1 / 50  # 50 Hz like Aloha

    for fn in [time.sleep, precise_sleep]:
        times = []
        for _ in trange(1000):
            start = time.perf_counter()
            fn(sleep_time)
            times.append(time.perf_counter() - start)
        print(fn.__name__)
        print("Max time:", max(times))
        print("Min time:", min(times))
        mean_time = sum(times) / len(times)
        print("Mean:", mean_time)
        print("Std:", math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times)))
        print("===")
