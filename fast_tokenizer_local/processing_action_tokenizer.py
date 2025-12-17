import logging
from typing import ClassVar

import numpy as np
from scipy.fft import dct
from scipy.fft import idct
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from transformers.processing_utils import ProcessorMixin


class UniversalActionProcessor(ProcessorMixin):
    attributes: ClassVar[list[str]] = ["bpe_tokenizer"]
    bpe_tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        bpe_tokenizer: PreTrainedTokenizerFast,
        scale: float = 10,
        vocab_size: int = 1024,
        min_token: int = 0,
        *,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ):
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token

        # Action horizon and dimension needed during decoding. These can be specified
        # in three ways (in order of priority):
        # 1. passed in as kwargs to decode()
        # 2. in the constructor
        # 3. cached from the last time decode() was called
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

        super().__init__(bpe_tokenizer)

    def __call__(self, action_chunk: np.array) -> np.array:
        assert action_chunk.ndim <= 3, "Only 3 dimensions supported: [batch, timesteps, action_dim]"
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]

        # Cache the time horizon and action dimension for decoding
        self.called_time_horizon = action_chunk.shape[-2]
        self.called_action_dim = action_chunk.shape[-1]

        dct_coeff = dct(action_chunk, axis=1, norm="ortho")
        dct_coeff = np.around(dct_coeff * self.scale)
        tokens = []
        for elem in dct_coeff:
            token_str = "".join(map(chr, np.maximum(elem.flatten() - self.min_token, 0).astype(int)))
            tokens.append(self.bpe_tokenizer(token_str)["input_ids"])
        return tokens

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.array:
        self.time_horizon = time_horizon or self.time_horizon or self.called_time_horizon
        self.action_dim = action_dim or self.action_dim or self.called_action_dim

        # Cache the time horizon and action dimension for the next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim

        assert (
            self.time_horizon is not None and self.action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        for token in tokens:
            try:
                decoded_tokens = self.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.min_token
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self.action_dim)
                assert (
                    decoded_dct_coeff.shape
                    == (
                        self.time_horizon,
                        self.action_dim,
                    )
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({self.time_horizon}, {self.action_dim})"
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                print(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((self.time_horizon, self.action_dim))
            decoded_actions.append(idct(decoded_dct_coeff / self.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions)

    @classmethod
    def fit(
        cls,
        action_data: list[np.array],
        scale: float = 10,
        vocab_size: int = 1024,
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> "UniversalActionProcessor":
        # Run DCT over all inputs
        dct_tokens = [dct(a, axis=0, norm="ortho").flatten() for a in action_data]

        # Quantize and find min token
        max_token = int(np.around(np.concatenate(dct_tokens) * scale).max())
        min_token = int(np.around(np.concatenate(dct_tokens) * scale).min())
        min_vocab_size = max_token - min_token

        assert (
            min_vocab_size <= vocab_size
        ), f"Vocab size {vocab_size} is too small for the range of tokens {min_vocab_size}"
        if min_vocab_size + 100 > vocab_size:
            logging.warning(
                f"Initial alphabet size {min_vocab_size} is almost as large as the vocab"
                f"size {vocab_size}, consider increasing vocab size"
            )

        # Make token iterator for BPE training
        def _token_iter():
            for tokens in dct_tokens:
                rounded_tokens = np.around(tokens * scale) - min_token
                rounded_tokens = rounded_tokens.astype(int)
                string = "".join(map(chr, rounded_tokens))
                yield string

        # Train BPE tokenizer
        bpe = ByteLevelBPETokenizer()

        # Set up the entire range of possible tokens as the initial alphabet
        alphabet = [chr(i) for i in range(max_token - min_token + 1)]
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[],
            initial_alphabet=alphabet,
            max_token_length=10000,
        )

        # Train the inner tokenizer (don't use ByteLevelBPETokenizer.train_from_iterator()
        # because it doesn't support custom alphabets)
        bpe._tokenizer.train_from_iterator(_token_iter(), trainer=trainer)

        return cls(
            PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False),
            scale=scale,
            vocab_size=vocab_size,
            min_token=min_token,
            time_horizon=time_horizon,
            action_dim=action_dim,
        )
