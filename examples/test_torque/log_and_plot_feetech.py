

import matplotlib.pyplot as plt
import time
from typing import List, Tuple
def log_and_plot_params(bus, params_to_log: list, servo_names: list, 
                        test_id="servo_log", interval=0.1, duration=5, save_plot=True) -> Tuple[dict, List[float]]:
    
    """
    Logs specific servo parameters for a given duration and generates a plot.
    """

    servo_data = {servo_name: {param: [] for param in params_to_log} for servo_name in servo_names}
    timestamps = []

    start_time = time.time()

    while time.time() - start_time < duration:
        timestamp = time.time() - start_time
        timestamps.append(timestamp)
        for param in params_to_log:
            values = bus.read(param, servo_names)
            for servo_name, value in zip(servo_names, values):
                servo_data[servo_name][param].append(value)

        time.sleep(interval)

    if save_plot:
        for servo_name, data in servo_data.items():
            plt.figure(figsize=(10, 6))
            for param in params_to_log:
                if all(v is not None for v in data[param]):
                    plt.plot(timestamps, data[param], label=param)
            plt.xlabel("Time (s)")
            plt.ylabel("Parameter Values")
            plt.title(f"Parameter Trends for Servo: {servo_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_filename = f"{test_id}_{servo_name}.png"
            plt.savefig(plot_filename)
            print(f"Plot saved as {plot_filename}")
            
    return servo_data, timestamps
