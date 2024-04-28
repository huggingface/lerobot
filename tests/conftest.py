from .utils import DEVICE


def pytest_addoption(parser):
    parser.addoption(
        "--disable-data-loader-warning", action="store_true", help="Silence the DataLoader worker warning"
    )


def pytest_configure(config):
    config.disable_data_loader_warning = config.getoption("--disable-data-loader-warning")


def pytest_collection_finish(session):
    print(f"\nTesting with {DEVICE=}")
    if session.config.disable_data_loader_warning:
        import warnings

        warnings.filterwarnings(
            "ignore",
            message="This DataLoader will create .* worker processes in total. Our suggested max number of worker in current system is .*",
            category=UserWarning,
        )
