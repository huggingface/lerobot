import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Adjust this to match your actual serial port and baud rate
SERIAL_PORT = '/dev/ttyACM0'  # or COM3 on Windows
BAUD_RATE = 115200

# Set up serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

# How many data points to keep in the scrolling buffer
buffer_len = 200

# Create buffers for each sensor pair.
# We'll store them in a dict to keep things organized.
sensor_buffers = {
    'wrist_roll': {
        'val1': deque([0]*buffer_len, maxlen=buffer_len),
        'val2': deque([0]*buffer_len, maxlen=buffer_len)
    },
    'elbow_pitch': {
        'val1': deque([0]*buffer_len, maxlen=buffer_len),
        'val2': deque([0]*buffer_len, maxlen=buffer_len)
    },
    'shoulder_pitch': {
        'val1': deque([0]*buffer_len, maxlen=buffer_len),
        'val2': deque([0]*buffer_len, maxlen=buffer_len)
    },
    'shoulder_yaw': {
        'val1': deque([0]*buffer_len, maxlen=buffer_len),
        'val2': deque([0]*buffer_len, maxlen=buffer_len)
    },
    'shoulder_roll': {
        'val1': deque([0]*buffer_len, maxlen=buffer_len),
        'val2': deque([0]*buffer_len, maxlen=buffer_len)
    }
}

# Create a figure with 5 subplots (one for each sensor pair).
fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
fig.tight_layout(pad=3.0)

# We'll store line references in a dict so we can update them in our update() function.
lines = {
    'wrist_roll': [],
    'elbow_pitch': [],
    'shoulder_pitch': [],
    'shoulder_yaw': [],
    'shoulder_roll': []
}

# Set up each subplot
subplot_info = [
    ('wrist_roll',  'Wrist Roll (2,3)',  axes[0]),
    ('elbow_pitch', 'Elbow Pitch (0,1)', axes[1]),
    ('shoulder_pitch', 'Shoulder Pitch (10,11)', axes[2]),
    ('shoulder_yaw',   'Shoulder Yaw (12,13)',   axes[3]),
    ('shoulder_roll',  'Shoulder Roll (14,15)',  axes[4])
]

for (sensor_name, label, ax) in subplot_info:
    ax.set_title(label)
    ax.set_xlim(0, buffer_len)
    ax.set_ylim(0, 4096)
    line1, = ax.plot([], [], label=f"{sensor_name} - val1")
    line2, = ax.plot([], [], label=f"{sensor_name} - val2")
    ax.legend()
    lines[sensor_name] = [line1, line2]

def update(frame):
    # Read all available lines from the serial buffer
    while ser.in_waiting:
        raw_line = ser.readline().decode('utf-8').strip()
        parts = raw_line.split()
        
        # We expect at least 16 values if all sensors are present.
        # (Because you mentioned indices 0..1, 2..3, 10..11, 12..13, 14..15)
        if len(parts) < 16:
            continue
        
        try:
            values = list(map(int, parts))
        except ValueError:
            # If there's a parsing error, skip this line
            continue
        
        # Extract the relevant values and append to the correct buffer
        sensor_buffers['elbow_pitch']['val1'].append(values[0])
        sensor_buffers['elbow_pitch']['val2'].append(values[1])
        
        sensor_buffers['wrist_roll']['val1'].append(values[2])
        sensor_buffers['wrist_roll']['val2'].append(values[3])
        
        sensor_buffers['shoulder_pitch']['val1'].append(values[14])
        sensor_buffers['shoulder_pitch']['val2'].append(values[15])
        
        sensor_buffers['shoulder_yaw']['val1'].append(values[12])
        sensor_buffers['shoulder_yaw']['val2'].append(values[13])
        
        sensor_buffers['shoulder_roll']['val1'].append(values[10])
        sensor_buffers['shoulder_roll']['val2'].append(values[11])

    # Update each line's data in each subplot
    all_lines = []
    for (sensor_name, _, ax) in subplot_info:
        # x-values are just the index range of the buffer
        x_data = range(len(sensor_buffers[sensor_name]['val1']))
        
        # First line
        lines[sensor_name][0].set_data(
            x_data,
            sensor_buffers[sensor_name]['val1']
        )
        # Second line
        lines[sensor_name][1].set_data(
            x_data,
            sensor_buffers[sensor_name]['val2']
        )
        
        all_lines.extend(lines[sensor_name])
        
    return all_lines

# Create the animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)

plt.show()
