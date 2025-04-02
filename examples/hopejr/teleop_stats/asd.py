import serial
import threading
import time
import numpy as np
import matplotlib.pyplot as plt

# Thread function to read from a serial port continuously until stop_event is set.
def read_serial(port, baudrate, stop_event, data_list):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"Error opening {port}: {e}")
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().decode('utf-8').strip()
        except Exception as e:
            print(f"Decode error on {port}: {e}")
            continue

        if line:
            try:
                # Split the line into integer values.
                values = [int(x) for x in line.split()]
                # For ACM1, ignore the extra value if present.
                if len(values) >= 16:
                    if len(values) > 16:
                        values = values[:16]
                    # Save the timestamp (relative to start) with the sensor readings.
                    timestamp = time.time()
                    data_list.append((timestamp, values))
            except Exception as e:
                print(f"Error parsing line from {port}: '{line}' -> {e}")
    ser.close()

def main():
    # --- Configuration ---
    # Set your serial port names here (adjust for your system)
    acm0_port = "/dev/ttyACM0"  # Example for Linux (or "COM3" on Windows)
    acm1_port = "/dev/ttyACM1"  # Example for Linux (or "COM4" on Windows)
    baudrate = 115200

    # Data storage for each device:
    data_acm0 = []  # Will hold tuples of (timestamp, [16 sensor values])
    data_acm1 = []

    # Event to signal threads to stop reading.
    stop_event = threading.Event()

    # Create and start reader threads.
    thread_acm0 = threading.Thread(target=read_serial, args=(acm0_port, baudrate, stop_event, data_acm0))
    thread_acm1 = threading.Thread(target=read_serial, args=(acm1_port, baudrate, stop_event, data_acm1))
    thread_acm0.start()
    thread_acm1.start()

    # Record data for 10 seconds.
    record_duration = 10  # seconds
    start_time = time.time()
    time.sleep(record_duration)
    stop_event.set()  # signal threads to stop

    # Wait for both threads to finish.
    thread_acm0.join()
    thread_acm1.join()
    print("Finished recording.")

    # --- Process the Data ---
    # Convert lists of (timestamp, values) to numpy arrays.
    # Compute time relative to the start of the recording.
    times_acm0 = np.array([t - start_time for t, _ in data_acm0])
    sensor_acm0 = np.array([vals for _, vals in data_acm0])  # shape (N0, 16)

    times_acm1 = np.array([t - start_time for t, _ in data_acm1])
    sensor_acm1 = np.array([vals for _, vals in data_acm1])  # shape (N1, 16)

    # --- Plot 1: Overlapping Time Series ---
    plt.figure(figsize=(12, 8))
    # Plot each sensor from ACM0 in red.
    for i in range(16):
        plt.plot(times_acm0, sensor_acm0[:, i], color='red', alpha=0.7,
                 label='ACM0 Sensor 1' if i == 0 else None)
    # Plot each sensor from ACM1 in blue.
    for i in range(16):
        plt.plot(times_acm1, sensor_acm1[:, i], color='blue', alpha=0.7,
                 label='ACM1 Sensor 1' if i == 0 else None)
    plt.xlabel("Time (s)")
    plt.ylabel("Sensor Reading")
    plt.title("Overlapping Sensor Readings (ACM0 in Red, ACM1 in Blue)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("overlapping_sensor_readings.png", dpi=300)
    plt.close()
    print("Saved overlapping_sensor_readings.png")

    # --- Plot 2: Variance of Noise for Each Sensor ---
    # Compute variance (over time) for each sensor channel.
    variance_acm0 = np.var(sensor_acm0, axis=0)
    variance_acm1 = np.var(sensor_acm1, axis=0)
    sensor_numbers = np.arange(1, 17)
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(sensor_numbers - bar_width/2, variance_acm0, bar_width, color='red', label='ACM0')
    plt.bar(sensor_numbers + bar_width/2, variance_acm1, bar_width, color='blue', label='ACM1')
    plt.xlabel("Sensor Number")
    plt.ylabel("Variance")
    plt.title("Noise Variance per Sensor")
    plt.xticks(sensor_numbers)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sensor_variance.png", dpi=300)
    plt.close()
    print("Saved sensor_variance.png")

    # --- Plot 3: Difference Between ACM0 and ACM1 Readings ---
    # Since the two devices may not sample at exactly the same time,
    # we interpolate ACM1's data onto ACM0's time base for each sensor.
    plt.figure(figsize=(12, 8))
    for i in range(16):
        if len(times_acm1) > 1 and len(times_acm0) > 1:
            interp_acm1 = np.interp(times_acm0, times_acm1, sensor_acm1[:, i])
            diff = sensor_acm0[:, i] - interp_acm1
            plt.plot(times_acm0, diff, label=f"Sensor {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Difference (ACM0 - ACM1)")
    plt.title("Difference in Sensor Readings")
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("sensor_differences.png", dpi=300)
    plt.close()
    print("Saved sensor_differences.png")

if __name__ == "__main__":
    main()
