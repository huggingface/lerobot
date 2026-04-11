"""
Cross-cutting modules that bridge multiple lerobot packages.

Unlike ``lerobot.utils`` (which must remain dependency-free), modules here
are allowed to import from ``lerobot.policies``, ``lerobot.processor``,
``lerobot.configs``, etc.  They are deliberately NOT re-exported from the
top-level ``lerobot`` package.

Available modules (import directly)::

    from lerobot.common.control_utils import predict_action, ...
    from lerobot.common.train_utils import save_checkpoint, ...
    from lerobot.common.wandb_utils import WandBLogger, ...
"""

__all__: list[str] = []
