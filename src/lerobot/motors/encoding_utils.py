# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def encode_sign_magnitude(value: int, sign_bit_index: int):
    """Encode a signed integer using [sign-magnitude representation](https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude).

    Args:
        value (`int`):
            The signed value to encode.
        sign_bit_index (`int`):
            Index of the bit that carries the sign; bits below it carry the magnitude.

    Returns:
        `int`: The encoded value, with the sign bit set for negative inputs.

    Raises:
        ValueError: If `value`'s magnitude does not fit in `sign_bit_index` bits.

    Example:
        ```python
        >>> from lerobot.motors.encoding_utils import encode_sign_magnitude
        >>> encode_sign_magnitude(-5, sign_bit_index=7)
        133
        ```
    """
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(f"Magnitude {magnitude} exceeds {max_magnitude} (max for {sign_bit_index=})")

    direction_bit = 1 if value < 0 else 0
    return (direction_bit << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int):
    """Decode a [sign-magnitude](https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude)-encoded integer.

    The inverse of [`~motors.encoding_utils.encode_sign_magnitude`].

    Args:
        encoded_value (`int`):
            The encoded value to decode.
        sign_bit_index (`int`):
            Index of the bit that carries the sign; bits below it carry the magnitude.

    Returns:
        `int`: The decoded signed value.

    Example:
        ```python
        >>> from lerobot.motors.encoding_utils import decode_sign_magnitude
        >>> decode_sign_magnitude(133, sign_bit_index=7)
        -5
        ```
    """
    direction_bit = (encoded_value >> sign_bit_index) & 1
    magnitude_mask = (1 << sign_bit_index) - 1
    magnitude = encoded_value & magnitude_mask
    return -magnitude if direction_bit else magnitude


def encode_twos_complement(value: int, n_bytes: int):
    """Encode a signed integer using [two's-complement representation](https://en.wikipedia.org/wiki/Signed_number_representations#Two%27s_complement).

    Args:
        value (`int`):
            The signed value to encode.
        n_bytes (`int`):
            Width of the encoding, in bytes.

    Returns:
        `int`: The encoded value.

    Raises:
        ValueError: If `value` does not fit in `n_bytes` bytes.

    Example:
        ```python
        >>> from lerobot.motors.encoding_utils import encode_twos_complement
        >>> encode_twos_complement(-5, n_bytes=1)
        251
        ```
    """
    bit_width = n_bytes * 8
    min_val = -(1 << (bit_width - 1))
    max_val = (1 << (bit_width - 1)) - 1

    if not (min_val <= value <= max_val):
        raise ValueError(
            f"Value {value} out of range for {n_bytes}-byte two's complement: [{min_val}, {max_val}]"
        )

    if value >= 0:
        return value

    return (1 << bit_width) + value


def decode_twos_complement(value: int, n_bytes: int) -> int:
    """Decode a [two's-complement](https://en.wikipedia.org/wiki/Signed_number_representations#Two%27s_complement)-encoded integer.

    The inverse of [`~motors.encoding_utils.encode_twos_complement`].

    Args:
        value (`int`):
            The encoded value to decode.
        n_bytes (`int`):
            Width of the encoding, in bytes.

    Returns:
        `int`: The decoded signed value.

    Example:
        ```python
        >>> from lerobot.motors.encoding_utils import decode_twos_complement
        >>> decode_twos_complement(251, n_bytes=1)
        -5
        ```
    """
    bits = n_bytes * 8
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        value -= 1 << bits
    return value
