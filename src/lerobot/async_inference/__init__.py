# Async inference server/client.
# Requires: lerobot[async]

from lerobot.utils.import_utils import require_package

require_package("grpcio", extra="async", import_name="grpc")
