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
from collections import deque
from collections.abc import Iterable, Iterator


class LookBackError(Exception):
    """
    Exception raised when trying to look back in the history of a Backtrackable object.
    """

    pass


class LookAheadError(Exception):
    """
    Exception raised when trying to look ahead in the future of a Backtrackable object.
    """

    pass


class Backtrackable[T]:
    """
    Wrap any iterator/iterable so you can step back up to `history` items
    and look ahead up to `lookahead` items.

    This is useful for streaming datasets where you need to access previous and future items
    but can't load the entire dataset into memory.

    Example:
    -------
    ```python
    ds = load_dataset("c4", "en", streaming=True, split="train")
    rev = Backtrackable(ds, history=3, lookahead=2)

    x0 = next(rev)  # forward
    x1 = next(rev)
    x2 = next(rev)

    # Look ahead
    x3_peek = rev.peek_ahead(1)  # next item without moving cursor
    x4_peek = rev.peek_ahead(2)  # two items ahead

    # Look back
    x1_again = rev.peek_back(1)  # previous item without moving cursor
    x0_again = rev.peek_back(2)  # two items back

    # Move backward
    x1_back = rev.prev()  # back one step
    next(rev)  # returns x2, continues forward from where we were
    ```
    """

    __slots__ = ("_source", "_back_buf", "_ahead_buf", "_cursor", "_history", "_lookahead")

    def __init__(self, iterable: Iterable[T], *, history: int = 1, lookahead: int = 0):
        if history < 1:
            raise ValueError("history must be >= 1")
        if lookahead <= 0:
            raise ValueError("lookahead must be > 0")

        self._source: Iterator[T] = iter(iterable)
        self._back_buf: deque[T] = deque(maxlen=history)
        self._ahead_buf: deque[T] = deque(maxlen=lookahead) if lookahead > 0 else deque()
        self._cursor: int = 0
        self._history = history
        self._lookahead = lookahead

    def __iter__(self) -> "Backtrackable[T]":
        return self

    def __next__(self) -> T:
        # If we've stepped back, consume from back buffer first
        if self._cursor < 0:  # -1 means "last item", etc.
            self._cursor += 1
            return self._back_buf[self._cursor]

        # If we have items in the ahead buffer, use them first
        item = self._ahead_buf.popleft() if self._ahead_buf else next(self._source)

        # Add current item to back buffer and reset cursor
        self._back_buf.append(item)
        self._cursor = 0
        return item

    def prev(self) -> T:
        """
        Step one item back in history and return it.
        Raises IndexError if already at the oldest buffered item.
        """
        if len(self._back_buf) + self._cursor <= 1:
            raise LookBackError("At start of history")

        self._cursor -= 1
        return self._back_buf[self._cursor]

    def peek_back(self, n: int = 1) -> T:
        """
        Look `n` items back (n=1 == previous item) without moving the cursor.
        """
        if n < 0 or n + 1 > len(self._back_buf) + self._cursor:
            raise LookBackError("peek_back distance out of range")

        return self._back_buf[self._cursor - (n + 1)]

    def peek_ahead(self, n: int = 1) -> T:
        """
        Look `n` items ahead (n=1 == next item) without moving the cursor.
        Fills the ahead buffer if necessary.
        """
        if n < 1:
            raise LookAheadError("peek_ahead distance must be 1 or more")
        elif n > self._lookahead:
            raise LookAheadError("peek_ahead distance exceeds lookahead limit")

        # Fill ahead buffer if we don't have enough items
        while len(self._ahead_buf) < n:
            try:
                item = next(self._source)
                self._ahead_buf.append(item)

            except StopIteration as err:
                raise LookAheadError("peek_ahead: not enough items in source") from err

        return self._ahead_buf[n - 1]

    def history(self) -> list[T]:
        """
        Return a copy of the buffered history (most recent last).
        The list length ≤ `history` argument passed at construction.
        """
        if self._cursor == 0:
            return list(self._back_buf)

        # When cursor<0, slice so the order remains chronological
        return list(self._back_buf)[: self._cursor or None]

    def can_peek_back(self, steps: int = 1) -> bool:
        """
        Check if we can go back `steps` items without raising an IndexError.
        """
        return steps <= len(self._back_buf) + self._cursor

    def can_peek_ahead(self, steps: int = 1) -> bool:
        """
        Check if we can peek ahead `steps` items.
        This may involve trying to fill the ahead buffer.
        """
        if self._lookahead > 0 and steps > self._lookahead:
            return False

        # Try to fill ahead buffer to check if we can peek that far
        try:
            while len(self._ahead_buf) < steps:
                if self._lookahead > 0 and len(self._ahead_buf) >= self._lookahead:
                    return False
                item = next(self._source)
                self._ahead_buf.append(item)
            return True
        except StopIteration:
            return False
