# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Portable trusted configuration bundles; not executable code or credentials."""

from urllib.parse import urlsplit

from .types import fingerprint


def check_credentials(value):
    """Reject literal secrets and credential-bearing URLs in persistent records."""
    if isinstance(value, dict):
        for key, entry in value.items():
            if key.lower() in {"token", "password", "secret", "credentials", "api_key"} and entry not in (
                None,
                "",
                "EMPTY",
            ):
                raise ValueError(f"Use an environment reference rather than a literal {key}")
            check_credentials(entry)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            check_credentials(entry)
    elif isinstance(value, str) and "://" in value:
        url = urlsplit(value)
        if url.username or url.password or url.query:
            raise ValueError("Persistent URLs cannot contain credentials or signed query strings")


def write_bundle(store, action, config, code_revision):
    bundle = {"version": 1, "action": action, "code_revision": code_revision, "config": config}
    check_credentials(bundle)
    digest = fingerprint(bundle)
    key = f"bundles/{digest}.json"
    if store.exists(key):
        if store.checksum(key)[0] != digest:
            raise ValueError("Existing processing bundle checksum mismatch")
    else:
        store.put_json(key, bundle)
    return key, digest
