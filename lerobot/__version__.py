"""To enable `lerobot.__version__`"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lerobot")
except PackageNotFoundError:
    __version__ = "unknown"
