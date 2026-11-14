# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from scipy.fft import dct, idct  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from lerobot.policies.fineart_vla.fit_fast_tokenizer import _complete_byte_alphabet  # noqa: E402


def test_fitted_fast_preserves_unseen_utf8_bytes_and_action_coefficients():
    rng = np.random.default_rng(42)
    actions = rng.normal(size=(64, 50, 7))
    held_out = rng.normal(size=(16, 50, 7))
    coefficients = np.around(dct(actions, axis=1, norm="ortho") * 10)
    minimum = int(coefficients.min())
    texts = ["".join(map(chr, (row.flatten() - minimum).astype(int))) for row in coefficients]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    # Reproduce the upstream fitter's incomplete observed-character alphabet.
    backend.train_from_iterator(
        texts, trainer=trainers.BpeTrainer(vocab_size=512, initial_alphabet=[chr(i) for i in range(80)])
    )
    fitted = SimpleNamespace(
        scale=10,
        min_token=minimum,
        vocab_size=512,
        bpe_tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend),
    )
    _complete_byte_alphabet(fitted, actions)
    tokenizer = fitted.bpe_tokenizer
    assert tokenizer.decode(tokenizer.encode("€🙂")) == "€🙂"
    decoded = []
    for row in np.around(dct(held_out, axis=1, norm="ortho") * fitted.scale):
        text = "".join(map(chr, np.maximum(row.flatten() - minimum, 0).astype(int)))
        restored = tokenizer.decode(tokenizer.encode(text))
        assert restored == text
        coeff = (np.array(list(map(ord, restored))) + minimum).reshape(50, 7)
        decoded.append(idct(coeff / fitted.scale, axis=0, norm="ortho"))
    assert np.sqrt(np.mean((np.array(decoded) - held_out) ** 2)) < 0.04
