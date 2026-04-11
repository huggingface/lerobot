# Reinforcement learning modules.
# Requires: lerobot[hilserl]

from lerobot.utils.import_utils import require_package

require_package("grpcio", extra="hilserl", import_name="grpc")
