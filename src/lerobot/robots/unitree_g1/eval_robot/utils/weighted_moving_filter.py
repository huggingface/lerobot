import numpy as np
import matplotlib.pyplot as plt


class WeightedMovingFilter:
    def __init__(self, weights, data_size=14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        # assert np.isclose(np.sum(self._weights), 1.0), "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        temp_filtered_data = np.zeros(self._data_size)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(data_array[:, i], self._weights, mode="valid")[-1]

        return temp_filtered_data

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # skip duplicate data

        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data


def visualize_filter_comparison(filter_params, steps):
    import time

    t = np.linspace(0, 4 * np.pi, steps)
    original_data = np.array(
        [np.sin(t + i) + np.random.normal(0, 0.2, len(t)) for i in range(35)]
    ).T  # sin wave with noise, shape is [len(t), 35]

    plt.figure(figsize=(14, 10))

    for idx, weights in enumerate(filter_params):
        filter = WeightedMovingFilter(weights, 14)
        data_2b_filtered = original_data.copy()
        filtered_data = []

        time1 = time.time()

        for i in range(steps):
            filter.add_data(data_2b_filtered[i][13:27])  # step i, columns 13 to 26 (total:14)
            data_2b_filtered[i][13:27] = filter.filtered_data
            filtered_data.append(data_2b_filtered[i])

        time2 = time.time()
        print(f"filter_params:{filter_params[idx]}, time cosume:{time2 - time1}")

        filtered_data = np.array(filtered_data)

        # col0 should not 2b filtered
        plt.subplot(len(filter_params), 2, idx * 2 + 1)
        plt.plot(filtered_data[:, 0], label=f"Filtered (Window {filter._window_size})")
        plt.plot(original_data[:, 0], "r--", label="Original", alpha=0.5)
        plt.title("Joint 1 - Should not to be filtered.")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()

        # col13 should 2b filtered
        plt.subplot(len(filter_params), 2, idx * 2 + 2)
        plt.plot(filtered_data[:, 13], label=f"Filtered (Window {filter._window_size})")
        plt.plot(original_data[:, 13], "r--", label="Original", alpha=0.5)
        plt.title(f"Joint 13 - Window {filter._window_size}, Weights {weights}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # windows_size and weights
    filter_params = [
        (np.array([0.7, 0.2, 0.1])),
        (np.array([0.5, 0.3, 0.2])),
        (np.array([0.4, 0.3, 0.2, 0.1])),
    ]

    visualize_filter_comparison(filter_params, steps=100)
