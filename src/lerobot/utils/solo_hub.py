"""
Solo Hub integration helpers for LeRobot.

Provides a bridge between LeRobot's from_pretrained pattern and the Solo Hub
client in solo-cli. The solo-cli package is lazily imported to avoid a hard
dependency -- LeRobot continues to work with HuggingFace Hub when solo-cli
is not installed.

Solo Hub model references use the 'solo:' prefix:
    solo:org/model_name
    solo://org/model_name
"""

SOLO_PREFIX = "solo:"
SOLO_URL_PREFIX = "solo://"


def is_solo_ref(identifier: str) -> bool:
    """Check if identifier uses solo: or solo:// prefix."""
    s = str(identifier)
    return s.startswith(SOLO_PREFIX) or s.startswith(SOLO_URL_PREFIX)


def parse_solo_ref(identifier: str) -> str:
    """Strip solo: prefix, returning org/model_name."""
    s = str(identifier)
    if s.startswith(SOLO_URL_PREFIX):
        return s[len(SOLO_URL_PREFIX):]
    if s.startswith(SOLO_PREFIX):
        return s[len(SOLO_PREFIX):]
    return s


def _check_solo_cli():
    """Verify solo-cli is installed, raise helpful ImportError if not."""
    try:
        import solo.hub  # noqa: F401
    except ImportError:
        raise ImportError(
            "Solo Hub support requires the solo-cli package. "
            "Install it with: pip install solo-cli"
        )


def solo_hub_download(repo_id: str, filename: str, **kwargs) -> str:
    """
    Download a single file from Solo Hub.
    Delegates to solo.hub.solo_hub_download.

    Args:
        repo_id: Model identifier as 'org/model_name' (without solo: prefix).
        filename: The file to download.
        **kwargs: Passed through (force_download, cache_dir, token, revision).

    Returns:
        Local path to the downloaded file.
    """
    _check_solo_cli()
    from solo.hub import solo_hub_download as _download
    return _download(repo_id=repo_id, filename=filename, **kwargs)


def solo_snapshot_download(repo_id: str, **kwargs) -> str:
    """
    Download all files for a model from Solo Hub.
    Delegates to solo.hub.solo_snapshot_download.

    Args:
        repo_id: Model identifier as 'org/model_name' (without solo: prefix).
        **kwargs: Passed through (allow_patterns, ignore_patterns, force_download, etc.).

    Returns:
        Local path to the snapshot directory.
    """
    _check_solo_cli()
    from solo.hub import solo_snapshot_download as _download
    return _download(repo_id=repo_id, **kwargs)
