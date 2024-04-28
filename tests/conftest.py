from .utils import DEVICE


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")
