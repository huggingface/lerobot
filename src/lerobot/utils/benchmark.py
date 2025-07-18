#!/usr/bin/env python

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
import threading
import time
from contextlib import ContextDecorator


class TimeBenchmark(ContextDecorator):
    """
    Measures execution time using a context manager or decorator.

    This class supports both context manager and decorator usage, and is thread-safe for multithreaded
    environments.

    Args:
        print: If True, prints the elapsed time upon exiting the context or completing the function. Defaults
        to False.

    Examples:

        Using as a context manager:

        >>> benchmark = TimeBenchmark()
        >>> with benchmark:
        ...     time.sleep(1)
        >>> print(f"Block took {benchmark.result:.4f} seconds")
        Block took approximately 1.0000 seconds

        Using with multithreading:

        ```python
        import threading

        benchmark = TimeBenchmark()


        def context_manager_example():
            with benchmark:
                time.sleep(0.01)
            print(f"Block took {benchmark.result_ms:.2f} milliseconds")


        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=context_manager_example)
            threads.append(t1)

        for t in threads:
            t.start()

        for t in threads:
            t.join()
        ```
        Expected output:
        Block took approximately 10.00 milliseconds
        Block took approximately 10.00 milliseconds
        Block took approximately 10.00 milliseconds
    """

    def __init__(self, print=False):
        self.local = threading.local()
        self.print_time = print

    def __enter__(self):
        self.local.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.local.end_time = time.perf_counter()
        self.local.elapsed_time = self.local.end_time - self.local.start_time
        if self.print_time:
            print(f"Elapsed time: {self.local.elapsed_time:.4f} seconds")
        return False

    @property
    def result(self):
        return getattr(self.local, "elapsed_time", None)

    @property
    def result_ms(self):
        return self.result * 1e3
