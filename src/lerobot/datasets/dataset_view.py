# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Lazy, read-only column views for storage-specific dataset readers."""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
from datasets import Features
from datasets.formatting import get_formatter

if TYPE_CHECKING:
    import datasets

    from .dataset_reader import BaseDatasetReader


class DatasetColumnView:
    """A projected dataset supporting indexing, iteration and HF read formats.

    The view keeps its reader's episode selection and reads only the requested
    columns. It does not expand temporal windows or apply image augmentations.
    It implements the read surface used by LeRobot without materializing a
    Hugging Face Dataset or exposing its editing methods.
    """

    def __init__(
        self,
        reader: BaseDatasetReader,
        column_names: str | list[str],
        format_type: str | None = "custom",
        format_kwargs: dict[str, Any] | None = None,
    ) -> None:
        columns = [column_names] if isinstance(column_names, str) else list(column_names)
        missing = set(columns) - reader.read_features.keys()
        if missing:
            raise ValueError(f"Columns {sorted(missing)} not in the dataset: {list(reader.read_features)}")
        if len(set(columns)) != len(columns):
            raise ValueError("Selected columns must be unique.")
        self._reader = reader
        self._columns = columns
        self._format_type = format_type
        self._format_kwargs = dict(format_kwargs or {})

    @property
    def column_names(self) -> list[str]:
        """Names of the columns exposed by this view."""
        return list(self._columns)

    @property
    def features(self) -> datasets.Features:
        """HF feature definitions of the selected columns."""
        features = self._reader.read_features.copy()
        return Features({key: features[key] for key in self._columns})

    def __len__(self) -> int:
        """Number of frames in the reader's episode selection."""
        return len(self._reader) if self._columns else 0

    @overload
    def __getitem__(self, key: str) -> DatasetColumn: ...

    @overload
    def __getitem__(self, key: int | slice | Sequence[int]) -> Any: ...

    def __getitem__(self, key: str | int | slice | Sequence[int]) -> Any:
        """Read one row, a row batch, or a lazy column."""
        if isinstance(key, str):
            if key not in self._columns:
                raise KeyError(key)
            return DatasetColumn(self.select_columns(key))
        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self))))
            query_type = "batch"
        elif isinstance(key, Sequence | np.ndarray):
            indices = [operator.index(index) for index in cast(Sequence[int], key)]
            query_type = "batch"
        else:
            indices = [operator.index(key)]
            query_type = "row"
        if any(index < -len(self) or index >= len(self) for index in indices):
            raise IndexError(f"Row index is out of bounds for a view with {len(self)} rows.")
        return self._reader.read_rows(
            indices,
            self._columns,
            query_type=query_type,
            format_type=self._format_type,
            **self._format_kwargs,
        )

    def __iter__(self) -> Iterator[Any]:
        """Iterate over rows with bounded batch reads."""
        if self._format_type in {"arrow", "pandas", "polars"}:
            for index in range(len(self)):
                yield self[index]
            return
        for batch in self.iter(batch_size=512):
            size = len(batch[self._columns[0]])
            for index in range(size):
                yield {key: values[index] for key, values in batch.items()}

    def iter(self, batch_size: int = 512) -> Iterator[Any]:
        """Iterate over batches of the requested size."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(self), batch_size):
            yield self[start : start + batch_size]

    def select_columns(self, column_names: str | list[str]) -> DatasetColumnView:
        """Return a narrower view without reading any rows."""
        columns = [column_names] if isinstance(column_names, str) else list(column_names)
        missing = set(columns) - set(self._columns)
        if missing:
            raise ValueError(f"Columns {sorted(missing)} not in the selected view.")
        return DatasetColumnView(self._reader, columns, self._format_type, self._format_kwargs)

    def with_format(self, type: str | None = None, **format_kwargs: Any) -> DatasetColumnView:
        """Return a view using a HF format such as Python, NumPy, Torch or Arrow."""
        get_formatter(type, features=self.features, **format_kwargs)
        return DatasetColumnView(self._reader, self._columns, type, format_kwargs)


class DatasetColumn:
    """A lazy sequence exposing one projected column."""

    def __init__(self, view: DatasetColumnView) -> None:
        self._view = view

    def __len__(self) -> int:
        """Number of values in the selected episodes."""
        return len(self._view)

    def __getitem__(self, key: int | slice | Sequence[int]) -> Any:
        """Read one value or a batch of values."""
        return self._view[key][self._view.column_names[0]]

    def __iter__(self) -> Iterator[Any]:
        """Read values in bounded batches."""
        name = self._view.column_names[0]
        for batch in self._view.iter(batch_size=512):
            yield from batch[name]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Materialize this column when NumPy explicitly requests an array."""
        array = np.asarray(self[:], dtype=dtype)
        return array.copy() if copy else array

    def tolist(self) -> list[Any]:
        """Read the column as a Python list."""
        return list(self)
