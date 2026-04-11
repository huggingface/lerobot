"""
Async inference server/client.

Requires: ``pip install 'lerobot[async]'``

Available modules (import directly)::

    from lerobot.async_inference.policy_server import ...
    from lerobot.async_inference.robot_client import ...
"""

from lerobot.utils.import_utils import require_package

require_package("grpcio", extra="async", import_name="grpc")

__all__: list[str] = []
