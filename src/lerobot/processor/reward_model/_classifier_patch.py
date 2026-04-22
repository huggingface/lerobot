"""Monkey-patch for upstream ``Classifier.predict_reward``.

Upstream ``Classifier.predict_reward`` calls ``self.normalize_inputs`` /
``self.normalize_targets``, but classifier normalization was migrated into
a separate ``PolicyProcessorPipeline`` saved alongside the checkpoint as
``classifier_preprocessor.json`` + ``.safetensors``. As a result, every
inference call raises ``AttributeError: 'Classifier' object has no attribute
'normalize_inputs'``.

This patch installs a replacement ``predict_reward`` that:
1. Lazily loads ``classifier_preprocessor`` from the checkpoint dir and caches it.
2. Applies the pipeline to the batch (mean/std image normalization matching training).
3. Extracts normalized image tensors in ``input_features`` order.
4. Calls ``self.predict`` directly.
5. Returns a binary decision against ``threshold`` (for 2-class models).

It also patches ``Classifier.__init__`` to accept (and ignore) extra kwargs
(like ``dataset_stats``) that ``make_policy`` passes but upstream doesn't
expect.

Importing ``lerobot.processor.reward_model`` applies the patch once.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _load_preprocessor(classifier) -> Any | None:
    pretrained_path = getattr(classifier.config, "pretrained_path", None)
    if pretrained_path is None:
        logger.warning(
            "Classifier has no config.pretrained_path — cannot load "
            "classifier_preprocessor; predictions may be miscalibrated."
        )
        return None
    from lerobot.processor import DataProcessorPipeline

    try:
        return DataProcessorPipeline.from_pretrained(
            str(pretrained_path),
            config_filename="classifier_preprocessor.json",
        )
    except Exception as e:
        logger.warning(
            "Could not load classifier_preprocessor from %s: %s. "
            "Falling back to unnormalized inputs — predictions may be wrong.",
            pretrained_path,
            e,
        )
        return None


def _patched_predict_reward(self, batch, threshold: float = 0.5):
    from lerobot.utils.constants import OBS_IMAGE

    if not hasattr(self, "_cached_preprocessor"):
        self._cached_preprocessor = _load_preprocessor(self)

    if self._cached_preprocessor is not None:
        batch = self._cached_preprocessor(batch)

    target_device = next(self.parameters()).device

    images = []
    for key in self.config.input_features:
        if not key.startswith(OBS_IMAGE):
            continue
        img = batch[key]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        images.append(img.to(target_device))

    with torch.inference_mode():
        output = self.predict(images)

    probs = output.probabilities
    if self.config.num_classes == 2:
        return (probs > threshold).float()
    return torch.argmax(probs, dim=1)


def apply() -> None:
    """Install patches on ``Classifier`` (idempotent).

    1. ``predict_reward`` — use the external classifier_preprocessor pipeline.
    2. ``__init__`` — accept (and ignore) extra kwargs like ``dataset_stats``.
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
    except ImportError as e:
        logger.debug("Classifier not importable (%s) — skipping patch.", e)
        return

    _original_init = Classifier.__init__

    def _patched_init(self, config, **kwargs):
        _original_init(self, config)

    Classifier.__init__ = _patched_init  # type: ignore[method-assign]
    Classifier.predict_reward = _patched_predict_reward  # type: ignore[method-assign]
    _PATCHED = True
    logger.info("Patched Classifier.__init__ (accept extra kwargs) and predict_reward")
