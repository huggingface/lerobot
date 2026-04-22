"""Wrapper around ``lerobot-train`` that applies the Classifier patch first.

Upstream ``Classifier.__init__`` does not accept the ``dataset_stats`` kwarg
that ``make_policy`` passes to it, and ``predict_reward`` references removed
``normalize_inputs`` / ``normalize_targets`` attributes. Importing
``lerobot.processor.reward_model`` installs the patch; invoking
``lerobot-train`` directly skips that import, so the patch never runs and
training crashes.

This wrapper imports the patch, then calls ``lerobot_train.main``.

Usage (identical args to ``lerobot-train``):
    lerobot-train-reward-classifier --config_path=path/to/cfg.yaml
"""

from __future__ import annotations

import lerobot.processor.reward_model  # noqa: F401  — applies Classifier patch

from lerobot.scripts.lerobot_train import main


if __name__ == "__main__":
    main()
