import logging
import time

import numpy as np
import websockets.sync.client

import lerobot.common.utils.msgpack_utils as msgpack_utils

input = {
    "state": np.ones((13,)),
    "images": {
        # input images from client has spec h w c (client)
        "front": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "wrist_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
    },
    "prompt": "do something",
}

url = "ws://127.0.0.1:8000"
packer = msgpack_utils.Packer()

logging.info(f"Waiting for server at {url}...")
while True:
    try:
        conn = websockets.sync.client.connect(url, compression=None, max_size=None)
        metadata = msgpack_utils.unpackb(conn.recv())
        break
    except ConnectionRefusedError:
        logging.info("Still waiting for server...")
        time.sleep(5)
        
data = packer.pack(input)
conn.send(data)
response = conn.recv()
if isinstance(response, str):
    # we're expecting bytes; if the server sends a string, it's an error.
    print(f"Error in inference server:\n{response}")
    exit()

infer_result = msgpack_utils.unpackb(response)
print(infer_result)
assert len(infer_result['actions'][0]) == len(input['state'])

