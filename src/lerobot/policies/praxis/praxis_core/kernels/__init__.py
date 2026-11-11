# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/kernels/__init__.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Custom GPU kernels.

Every kernel in this folder is a custom kernel, not a PyTorch built-in.
Each file states which model part uses it and why.
"""
