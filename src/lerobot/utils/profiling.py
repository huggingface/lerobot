"""
Profiling utilities for performance analysis.

Usage:
    from lerobot.utils.profiling import profile_method, get_profiling_stats, print_profiling_summary

    @profile_method
    def my_slow_function(x):
        return x * 2

    # At end of execution:
    print_profiling_summary()
"""

import functools
import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Global profiling statistics storage
_profiling_stats: dict[str, list[float]] = defaultdict(list)
_profiling_lock = Lock()
_profiling_enabled = False


def enable_profiling():
    """Enable profiling globally."""
    global _profiling_enabled
    _profiling_enabled = True
    logger.info("Profiling enabled")


def disable_profiling():
    """Disable profiling globally."""
    global _profiling_enabled
    _profiling_enabled = False
    logger.info("Profiling disabled")


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled."""
    return _profiling_enabled


def record_timing(name: str, duration: float):
    """Record a timing measurement.

    Args:
        name: Name/identifier for this timing
        duration: Duration in seconds
    """
    if not _profiling_enabled:
        return

    with _profiling_lock:
        _profiling_stats[name].append(duration)


def profile_method(func: Callable) -> Callable:
    """Decorator to profile a method or function.

    Args:
        func: Function to profile

    Returns:
        Wrapped function that records execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if not _profiling_enabled:
            return func(*args, **kwargs)

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            # Use fully qualified name
            name = f"{func.__module__}.{func.__qualname__}"
            record_timing(name, duration)

    return wrapper


class ProfileContext:
    """Context manager for profiling code blocks.

    Usage:
        with ProfileContext("my_operation"):
            # ... code to profile ...
    """

    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        if _profiling_enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if _profiling_enabled and self.start is not None:
            duration = time.perf_counter() - self.start
            record_timing(self.name, duration)


def get_profiling_stats() -> dict[str, dict[str, float]]:
    """Get summary statistics for all profiled functions.

    Returns:
        Dictionary mapping function names to their stats (count, mean, min, max, total)
    """
    with _profiling_lock:
        summary = {}
        for name, times in _profiling_stats.items():
            if times:
                summary[name] = {
                    "count": len(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                    "mean_ms": (sum(times) / len(times)) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }
        return summary


def clear_profiling_stats():
    """Clear all profiling statistics."""
    with _profiling_lock:
        _profiling_stats.clear()
    logger.info("Profiling stats cleared")


def print_profiling_summary(sort_by: str = "total"):
    """Print formatted summary of profiling statistics.

    Args:
        sort_by: Sort key ('total', 'mean', 'count', 'max')
    """
    summary = get_profiling_stats()

    if not summary:
        logger.info("No profiling data available")
        return

    logger.info("\n" + "=" * 100)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 100)

    # Sort by requested key
    sorted_items = sorted(summary.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

    # Print header
    logger.info(
        f"{'Function':<60} {'Count':>8} {'Mean (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'Total (s)':>12}"
    )
    logger.info("-" * 100)

    # Print each function's stats
    for name, stats in sorted_items:
        # Shorten long names
        display_name = name if len(name) <= 60 else "..." + name[-57:]

        logger.info(
            f"{display_name:<60} "
            f"{stats['count']:>8} "
            f"{stats['mean_ms']:>12.2f} "
            f"{stats['min_ms']:>12.2f} "
            f"{stats['max_ms']:>12.2f} "
            f"{stats['total']:>12.2f}"
        )

    logger.info("=" * 100)

    # Print summary
    total_time = sum(s["total"] for s in summary.values())
    total_calls = sum(s["count"] for s in summary.values())
    logger.info(f"\nTotal profiled time: {total_time:.2f}s across {total_calls} calls")
    logger.info("=" * 100 + "\n")


def profile_section(name: str):
    """Return a context manager for profiling a code section.

    Args:
        name: Name for this section

    Returns:
        ProfileContext instance

    Usage:
        with profile_section("data_loading"):
            data = load_data()
    """
    return ProfileContext(name)

