# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Retry only positively identified temporary failures, not arbitrary exceptions."""

import errno
from enum import StrEnum
from urllib.error import HTTPError, URLError


class ErrorCategory(StrEnum):
    NETWORK = "temporary_network"
    SERVICE = "temporary_service"
    SCHEMA = "invalid_schema"
    INPUT = "undecodable_input"
    PERMANENT = "permanent_service"
    UNKNOWN = "unclassified"


class UndecodableInputError(ValueError):
    """A pinned input cannot be fully decoded; retries cannot repair its bytes."""


class TemporaryServiceError(RuntimeError):
    """An adapter explicitly identifies a temporary service failure."""


def classify_error(error: Exception) -> ErrorCategory:
    if isinstance(error, UndecodableInputError):
        return ErrorCategory.INPUT
    status = error.code if isinstance(error, HTTPError) else getattr(error, "status_code", None)
    if isinstance(status, int):
        return (
            ErrorCategory.SERVICE if status in {408, 429} or 500 <= status < 600 else ErrorCategory.PERMANENT
        )
    if isinstance(error, TemporaryServiceError):
        return ErrorCategory.SERVICE
    if isinstance(error, (TimeoutError, ConnectionError, URLError)):
        return ErrorCategory.NETWORK
    cls = type(error)
    if (
        cls.__module__.startswith("httpx")
        and cls.__name__
        in {
            "ConnectError",
            "ReadError",
            "WriteError",
            "RemoteProtocolError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
        }
    ) or (cls.__module__.startswith("openai") and cls.__name__ in {"APIConnectionError", "APITimeoutError"}):
        return ErrorCategory.NETWORK
    if isinstance(error, OSError) and error.errno in {
        errno.ETIMEDOUT,
        errno.ECONNRESET,
        errno.ECONNREFUSED,
        errno.EPIPE,
    }:
        return ErrorCategory.NETWORK
    # OpenAI/httpx wrap network exceptions. Follow the causal exception without
    # importing optional SDKs or classifying every SDK exception as retryable.
    cause = error.__cause__
    if isinstance(cause, Exception) and cause is not error and cause.__cause__ is not error:
        category = classify_error(cause)
        if category in {ErrorCategory.NETWORK, ErrorCategory.SERVICE}:
            return category
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return ErrorCategory.SCHEMA
    return ErrorCategory.UNKNOWN


def retry_delay(retry: int) -> float:
    """Bounded exponential backoff, independent of scientific item identity."""
    return min(30.0, 2.0 ** min(retry, 5))
