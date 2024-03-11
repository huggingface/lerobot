from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lerobot")
except PackageNotFoundError:
    __version__ = "unknown"
