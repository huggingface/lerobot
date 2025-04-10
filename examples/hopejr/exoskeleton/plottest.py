import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Adjust this to match your actual serial port and baud rate
SERIAL_PORT = '/dev/ttyACM0'  # or COM3 on Windows
BAUD_RATE = 115200

# Set up serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

# Buffers for real-time plot
buffer_len = 200
val1_buffer = deque([0]*buffer_len, maxlen=buffer_len)
val2_buffer = deque([0]*buffer_len, maxlen=buffer_len)

# Setup the plot
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Sensor 0')
line2, = ax.plot([], [], label='Sensor 1')
ax.set_ylim(0, 4096)
ax.set_xlim(0, buffer_len)
ax.legend()

def update(frame):
    while ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        parts = line.split()
        if len(parts) >= 2:
            try:
                val1 = int(parts[0])
                val2 = int(parts[1])
                val1_buffer.append(val1)
                val2_buffer.append(val2)
            except ValueError:
                pass  # skip malformed lines

    line1.set_ydata(val1_buffer)
    line1.set_xdata(range(len(val1_buffer)))
    line2.set_ydata(val2_buffer)
    line2.set_xdata(range(len(val2_buffer)))
    return line1, line2

ani = animation.FuncAnimation(fig, update, interval=50)
plt.show()
