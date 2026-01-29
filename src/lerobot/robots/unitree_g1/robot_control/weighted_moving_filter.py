"""
Weighted Moving Filter for smoothing IK solutions.
Ported from prometheus/src/xr_teleoperate/teleop/utils/weighted_moving_filter.py
"""

import numpy as np


class WeightedMovingFilter:
    """A weighted moving average filter for smoothing joint trajectories."""
    
    def __init__(self, weights: np.ndarray, data_size: int = 14):
        """
        Initialize the filter.
        
        Args:
            weights: Array of weights that sum to 1.0 (e.g., [0.4, 0.3, 0.2, 0.1])
            data_size: Number of dimensions in the data (e.g., 14 for arm joints)
        """
        self._window_size = len(weights)
        self._weights = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), \
            "[WeightedMovingFilter] the sum of weights must be 1.0!"
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self) -> np.ndarray:
        """Apply weighted moving average filter to queued data."""
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        temp_filtered_data = np.zeros(self._data_size)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(
                data_array[:, i], self._weights, mode='valid'
            )[-1]
        
        return temp_filtered_data

    def add_data(self, new_data: np.ndarray) -> None:
        """
        Add new data point and update filtered output.
        
        Args:
            new_data: Data array of shape (data_size,)
        """
        assert len(new_data) == self._data_size

        # Skip duplicate data
        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return
        
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self) -> np.ndarray:
        """Get the current filtered data."""
        return self._filtered_data
    
    def reset(self) -> None:
        """Reset the filter state."""
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []
