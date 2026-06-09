# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import numpy as np


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma
