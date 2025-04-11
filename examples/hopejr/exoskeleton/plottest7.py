import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Config
SERIAL_PORT = '/dev/ttyACM1'  # Change as needed
BAUD_RATE = 115200
BUFFER_LEN = 200

# Sensor names in order
sensor_names = [
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
    "elbow_flex",
    "shoulder_roll",
    "shoulder_yaw",
    "shoulder_pitch"
]

# Initialize buffers
sensor_data = {
    name: deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)
    for name in sensor_names
}

# Setup plot
fig, axes = plt.subplots(len(sensor_names), 1, figsize=(8, 12), sharex=True)
fig.tight_layout(pad=3.0)

lines = {}
for i, name in enumerate(sensor_names):
    axes[i].set_title(name)
    axes[i].set_xlim(0, BUFFER_LEN)
    axes[i].set_ylim(0, 4096)
    line, = axes[i].plot([], [], label=name)
    axes[i].legend()
    lines[name] = line

# Connect to serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

# Update function
def update(frame):
    while ser.in_waiting:
        line = ser.readline().decode().strip()
        parts = line.split()
        if len(parts) != 7:
            continue
        try:
            values = list(map(int, parts))
        except ValueError:
            continue
        for i, name in enumerate(sensor_names):
            sensor_data[name].append(values[i])
    for name in sensor_names:
        x = range(len(sensor_data[name]))
        lines[name].set_data(x, sensor_data[name])
    return lines.values()

# Animate
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
plt.show()
