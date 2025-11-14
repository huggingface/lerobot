"""Storage backend protocol for LeRobot datasets.

This module defines a minimal storage backend protocol that abstracts different
underlying formats (e.g., Parquet or Lance) behind a unified loading interface,
without disrupting the upper data pipeline.
"""
from pathlib import Path
from typing import Protocol

import datasets


class StorageBackendProtocol(Protocol):
    """Minimal read protocol.

    Implementers only need to provide a method that loads underlying data files
    as a Hugging Face Dataset.
    """

    def load_hf_dataset(
        self,
        root: Path,
        features: datasets.Features | None = None,
        episodes: list[int] | None = None,
    ) -> datasets.Dataset:
        """Load files under local root/data into a Hugging Face Dataset.

        Args:
            root: Dataset root directory containing data/, meta/, videos/, etc.
            features: Optional HF schema to ensure correct deserialization of complex columns.
            episodes: Optional episode indices. Current implementations typically load
                all files; this parameter is reserved for future selective filtering.
        Returns:
            Hugging Face datasets.Dataset
        """
        ...


def get_storage_backend(name: str, **kwargs):
    """Simple factory that returns a backend by name.

    Currently only "lance" is supported; other names raise ValueError.
    """
    if name is None:
        raise ValueError("storage backend name is None")
    if name.lower() == "lance":
        from .lance_backend import LanceBackend

        return LanceBackend(**kwargs)
    raise ValueError(f"Unsupported storage backend: {name}")
