import importlib
import logging


def is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
    Check if the package spec exists and grab its version to avoid importing a local directory.
    **Note:** this doesn't work for all packages.
    """
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logging.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_torch_available, _torch_version = is_package_available("torch", return_version=True)
_gym_xarm_available = is_package_available("gym_xarm")
_gym_aloha_available = is_package_available("gym_aloha")
_gym_pusht_available = is_package_available("gym_pusht")
