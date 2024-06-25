
# Defines a camera type
from typing import Protocol


class MotorsChain(Protocol):
    def write(self): ...
    def read(self): ...
