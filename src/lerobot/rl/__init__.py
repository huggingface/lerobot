"""
Reinforcement learning modules.

Requires: ``pip install 'lerobot[hilserl]'``

Available modules (import directly)::

    from lerobot.rl.actor import ...
    from lerobot.rl.learner import ...
    from lerobot.rl.learner_service import ...
    from lerobot.rl.buffer import ...
    from lerobot.rl.eval_policy import ...
    from lerobot.rl.gym_manipulator import ...
"""

from lerobot.utils.import_utils import require_package

require_package("grpcio", extra="hilserl", import_name="grpc")

__all__: list[str] = []
