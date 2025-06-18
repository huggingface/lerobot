import threading
import time

from mock_serial.mock_serial import Stub


class WaitableStub(Stub):
    """
    In some situations, a test might be checking if a stub has been called before `MockSerial` thread had time
    to read, match, and call the stub. In these situations, the test can fail randomly.

    Use `wait_called()` or `wait_calls()` to block until the stub is called, avoiding race conditions.

    Proposed fix:
    https://github.com/benthorner/mock_serial/pull/3
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event = threading.Event()

    def call(self):
        self._event.set()
        return super().call()

    def wait_called(self, timeout: float = 1.0):
        return self._event.wait(timeout)

    def wait_calls(self, min_calls: int = 1, timeout: float = 1.0):
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            if self.calls >= min_calls:
                return self.calls
            time.sleep(0.005)
        raise TimeoutError(f"Stub not called {min_calls} times within {timeout} seconds.")
