"""Minimal read implementation for the Lance storage backend.

This backend reads all Lance files under root/data/*/*.lance, concatenates them
into a PyArrow Table, then converts to a Hugging Face Dataset and sets
hf_transform_to_torch for downstream PyTorch use.

Dependencies are loaded on demand: lance is imported only when storage_backend="lance".
If lance is not installed, a clear ImportError is raised.
"""
from pathlib import Path
from typing import Any

import datasets
import pyarrow as pa

from lerobot.datasets.utils import hf_transform_to_torch


class LanceBackend:
    """Lance read backend.

    This implementation focuses on reading only. Writing/conversion is handled by
    src/lerobot/scripts/lerobot_convert_to_lance.py.
    """

    def __init__(self, params: dict | None = None):
        # Optional parameters reserved for future use, e.g., scan filters or batch size; currently unused
        self.params: dict[str, Any] = params or {}

    def _list_lance_files(self, root: Path) -> list[Path]:
        data_dir = root / "data"
        return sorted(data_dir.glob("*/*.lance"))

    def load_hf_dataset(
        self,
        root: Path,
        features: datasets.Features | None = None,
        episodes: list[int] | None = None,  # Episode-level filtering is not enabled for now to keep things simple
    ) -> datasets.Dataset:
        try:
            import lance  # type: ignore
        except Exception as e:  # ImportError or other
            raise ImportError(
                "storage_backend='lance' selected but the 'lance' package is not installed. Please `pip install lance`."
            ) from e

        lance_paths = self._list_lance_files(root)
        if len(lance_paths) == 0:
            raise FileNotFoundError(f"No .lance files found in: {root / 'data'}")

        # Read each file into an Arrow Table and concatenate
        tables: list[pa.Table] = []
        for p in lance_paths:
            ds = lance.dataset(str(p))  # Lance native dataset object
            # Convert to an Arrow Table; different versions may expose to_table() or to_arrow()
            if hasattr(ds, "to_table"):
                tbl = ds.to_table()
            elif hasattr(ds, "to_arrow"):
                tbl = ds.to_arrow()
            else:
                # Fallback: some versions expose a scanner to obtain a RecordBatchReader
                scanner = ds.to_scanner() if hasattr(ds, "to_scanner") else None
                if scanner is None:
                    raise RuntimeError("Cannot create an Arrow table from the Lance dataset. Please upgrade the 'lance' package.")
                tbl = scanner.to_table()
            tables.append(tbl)

        concat = pa.concat_tables(tables, promote=False)
        hf_ds = datasets.Dataset.from_arrow(concat, features=features, split="train")
        hf_ds.set_transform(hf_transform_to_torch)
        return hf_ds
