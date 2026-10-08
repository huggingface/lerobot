# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import importlib
from typing import Any


def load_source(factory: str, config: dict) -> Any:
    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("A source factory must be 'importable.module:ClassName'")
    return getattr(importlib.import_module(module), name)(**config)
