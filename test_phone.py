from teleop import Teleop
import threading

import time

count = 0
received = threading.Event()

def on_data(pose, message):
    global count
    count += 1
    print(f"[{count}] Got pose! Shape: {pose.shape}", flush=True)
    print(f"    Message: {message}", flush=True)
    received.set()

t = Teleop()
t.subscribe(on_data)
thread = threading.Thread(target=t.run, daemon=True)
thread.start()

print("Waiting for phone data...", flush=True)
print("Open the URL above in Chrome on your Android phone.", flush=True)
print("Tap 'Start', then touch and drag to stream pose data.\n", flush=True)

# Keep running and printing data until Ctrl+C
try:
    while True:
        time.sleep(1)
        if count > 0:
            print(f"  ... received {count} poses so far", flush=True)
        else:
            print("  ... still waiting for data", flush=True)
except KeyboardInterrupt:
    print(f"\nDone. Total poses received: {count}")
