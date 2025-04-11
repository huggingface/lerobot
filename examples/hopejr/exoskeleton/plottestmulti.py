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

# -------------------------------------------------------------------
# 1) Sensor buffers for existing sensors + new wrist_pitch, wrist_yaw
# -------------------------------------------------------------------
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
    },
    # --- New single-valued sensors ---
    'wrist_pitch': {
        'val1': deque([0]*buffer_len, maxlen=buffer_len)  # Only one line
    },
    'wrist_yaw': {
        'val1': deque([0]*buffer_len, maxlen=buffer_len)   # Only one line
    },
}

# -------------------------------------------------------------------
# 2) Figure with 7 subplots (was 5). We keep the original 5 + 2 new.
# -------------------------------------------------------------------
fig, axes = plt.subplots(7, 1, figsize=(8, 14), sharex=True)
fig.tight_layout(pad=3.0)

# We'll store line references in a dict so we can update them in update().
lines = {}

# -------------------------------------------------------------------
# 3) Define each subplot, including new ones at the end.
# -------------------------------------------------------------------
subplot_info = [
    ('wrist_roll',      'Wrist Roll (2,3)',        axes[0]),
    ('elbow_pitch',     'Elbow Pitch (0,1)',       axes[1]),
    ('shoulder_pitch',  'Shoulder Pitch (10,11)',  axes[2]),
    ('shoulder_yaw',    'Shoulder Yaw (12,13)',    axes[3]),
    ('shoulder_roll',   'Shoulder Roll (14,15)',   axes[4]),
    ('wrist_pitch',     'Wrist Pitch (0)',         axes[5]),  # new
    ('wrist_yaw',       'Wrist Yaw (1)',           axes[6]),  # new
]

# Set up each subplot
for (sensor_name, label, ax) in subplot_info:
    ax.set_title(label)
    ax.set_xlim(0, buffer_len)
    ax.set_ylim(0, 4096)  # adjust if needed
    
    # For existing sensors, plot 2 lines (val1, val2)
    # For the new single-line sensors, plot just 1 line
    if sensor_name in ['wrist_pitch', 'wrist_yaw']:
        # Single-valued
        line, = ax.plot([], [], label=f"{sensor_name}")
        lines[sensor_name] = line
    else:
        # Pair of values
        line1, = ax.plot([], [], label=f"{sensor_name} - val1")
        line2, = ax.plot([], [], label=f"{sensor_name} - val2")
        lines[sensor_name] = [line1, line2]
    
    ax.legend()

def update(frame):
    # Read all available lines from the serial buffer
    while ser.in_waiting:
        raw_line = ser.readline().decode('utf-8').strip()
        parts = raw_line.split()
        
        # We expect at least 16 values if all sensors are present
        if len(parts) < 7:
            continue
        
        try:
            values = list(map(int, parts))
        except ValueError:
            # If there's a parsing error, skip this line
            continue
        
        # Original code: extract the relevant values and append to the correct buffer
        sensor_buffers['elbow_pitch']['val1'].append(values[13])
        sensor_buffers['elbow_pitch']['val2'].append(values[13])
        
        sensor_buffers['wrist_roll']['val1'].append(values[3])
        sensor_buffers['wrist_roll']['val2'].append(values[3])
        
        sensor_buffers['shoulder_pitch']['val1'].append(values[14])
        sensor_buffers['shoulder_pitch']['val2'].append(values[14])
        
        sensor_buffers['shoulder_yaw']['val1'].append(values[8])
        sensor_buffers['shoulder_yaw']['val2'].append(values[8]) 
        
        sensor_buffers['shoulder_roll']['val1'].append(values[10])
        sensor_buffers['shoulder_roll']['val2'].append(values[10])
        
        # -------------------------------------------------------------------
        # 4) New code: also read wrist_pitch (index 0) and wrist_yaw (index 1)
        # -------------------------------------------------------------------
        sensor_buffers['wrist_yaw']['val1'].append(values[0])
        sensor_buffers['wrist_pitch']['val1'].append(values[1])

    # Update each line's data in each subplot
    all_lines = []
    for (sensor_name, _, ax) in subplot_info:
        # x-values are just the index range of the buffer for val1
        x_data = range(len(sensor_buffers[sensor_name]['val1']))
        
        # If this sensor has two lines
        if isinstance(lines[sensor_name], list):
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
        else:
            # Single line only (wrist_pitch, wrist_yaw)
            lines[sensor_name].set_data(
                x_data,
                sensor_buffers[sensor_name]['val1']
            )
            all_lines.append(lines[sensor_name])
        
    return all_lines

# Create the animation
ani = animation.FuncAnimation(fig, update, interval=50, blit=False)

plt.show()
