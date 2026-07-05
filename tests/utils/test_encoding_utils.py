#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import pytest

from lerobot.motors.encoding_utils import (
    decode_sign_magnitude,
    decode_twos_complement,
    encode_sign_magnitude,
    encode_twos_complement,
)


@pytest.mark.parametrize(
    "value, sign_bit_index, expected",
    [
        (5, 4, 5),
        (0, 4, 0),
        (7, 3, 7),
        (-1, 4, 17),
        (-8, 4, 24),
        (-3, 3, 11),
    ],
)
def test_encode_sign_magnitude(value, sign_bit_index, expected):
    assert encode_sign_magnitude(value, sign_bit_index) == expected


@pytest.mark.parametrize(
    "encoded, sign_bit_index, expected",
    [
        (5, 4, 5),
        (0, 4, 0),
        (7, 3, 7),
        (17, 4, -1),
        (24, 4, -8),
        (11, 3, -3),
    ],
)
def test_decode_sign_magnitude(encoded, sign_bit_index, expected):
    assert decode_sign_magnitude(encoded, sign_bit_index) == expected


@pytest.mark.parametrize(
    "encoded, sign_bit_index",
    [
        (16, 4),
        (-9, 3),
    ],
)
def test_encode_raises_on_overflow(encoded, sign_bit_index):
    with pytest.raises(ValueError):
        encode_sign_magnitude(encoded, sign_bit_index)


def test_encode_decode_sign_magnitude():
    for sign_bit_index in range(2, 6):
        max_val = (1 << sign_bit_index) - 1
        for value in range(-max_val, max_val + 1):
            encoded = encode_sign_magnitude(value, sign_bit_index)
            decoded = decode_sign_magnitude(encoded, sign_bit_index)
            assert decoded == value, f"Failed at value={value}, index={sign_bit_index}"


@pytest.mark.parametrize(
    "value, n_bytes, expected",
    [
        (0, 1, 0),
        (5, 1, 5),
        (-1, 1, 255),
        (-128, 1, 128),
        (-2, 1, 254),
        (127, 1, 127),
        (0, 2, 0),
        (5, 2, 5),
        (-1, 2, 65_535),
        (-32_768, 2, 32_768),
        (-2, 2, 65_534),
        (32_767, 2, 32_767),
        (0, 4, 0),
        (5, 4, 5),
        (-1, 4, 4_294_967_295),
        (-2_147_483_648, 4, 2_147_483_648),
        (-2, 4, 4_294_967_294),
        (2_147_483_647, 4, 2_147_483_647),
    ],
)
def test_encode_twos_complement(value, n_bytes, expected):
    assert encode_twos_complement(value, n_bytes) == expected


@pytest.mark.parametrize(
    "value, n_bytes, expected",
    [
        (0, 1, 0),
        (5, 1, 5),
        (255, 1, -1),
        (128, 1, -128),
        (254, 1, -2),
        (127, 1, 127),
        (0, 2, 0),
        (5, 2, 5),
        (65_535, 2, -1),
        (32_768, 2, -32_768),
        (65_534, 2, -2),
        (32_767, 2, 32_767),
        (0, 4, 0),
        (5, 4, 5),
        (4_294_967_295, 4, -1),
        (2_147_483_648, 4, -2_147_483_648),
        (4_294_967_294, 4, -2),
        (2_147_483_647, 4, 2_147_483_647),
    ],
)
def test_decode_twos_complement(value, n_bytes, expected):
    assert decode_twos_complement(value, n_bytes) == expected


@pytest.mark.parametrize(
    "value, n_bytes",
    [
        (-129, 1),
        (128, 1),
        (-32_769, 2),
        (32_768, 2),
        (-2_147_483_649, 4),
        (2_147_483_648, 4),
    ],
)
def test_encode_twos_complement_out_of_range(value, n_bytes):
    with pytest.raises(ValueError):
        encode_twos_complement(value, n_bytes)


@pytest.mark.parametrize(
    "value, n_bytes",
    [
        (-128, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (127, 1),
        (-32_768, 2),
        (-1, 2),
        (0, 2),
        (1, 2),
        (32_767, 2),
        (-2_147_483_648, 4),
        (-1, 4),
        (0, 4),
        (1, 4),
        (2_147_483_647, 4),
    ],
)
def test_encode_decode_twos_complement(value, n_bytes):
    encoded = encode_twos_complement(value, n_bytes)
    decoded = decode_twos_complement(encoded, n_bytes)
    assert decoded == value, f"Failed at value={value}, n_bytes={n_bytes}"
