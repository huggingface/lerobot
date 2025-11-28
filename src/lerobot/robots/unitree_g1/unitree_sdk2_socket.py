# unitree_sdk2_socket.py
import zmq
import pickle
import time

# you can tune these or read from env
ROBOT_IP = "172.18.129.215"
LOWCMD_PORT = 6000      # laptop -> robot
LOWSTATE_PORT = 6001    # robot -> laptop

_ctx = None
_lowcmd_sock = None
_lowstate_sock = None

def ChannelFactoryInitialize(*args, **kwargs):
    global _ctx, _lowcmd_sock, _lowstate_sock
    if _ctx is not None:
        return
    _ctx = zmq.Context.instance()

    # lowcmd: PUSH from laptop to robot
    _lowcmd_sock = _ctx.socket(zmq.PUSH)
    _lowcmd_sock.setsockopt(zmq.CONFLATE, 1)      
    _lowcmd_sock.connect(f"tcp://{ROBOT_IP}:{LOWCMD_PORT}")

    # lowstate: SUB from robot
    _lowstate_sock = _ctx.socket(zmq.SUB)      # no topic filtering
    _lowstate_sock.setsockopt(zmq.CONFLATE, 1)          # keep only last message
    _lowstate_sock.connect(f"tcp://{ROBOT_IP}:{LOWSTATE_PORT}")
    _lowstate_sock.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all


class ChannelPublisher:
    # just enough api for your code: __init__, Init, Write
    def __init__(self, topic, msg_type):
        # we ignore topic/msg_type, the bridge only supports the topics you use
        self.topic = topic
        self.msg_type = msg_type

    def Init(self):
        # nothing to do, sockets are global
        pass

    def Write(self, msg):
        # msg is hg_LowCmd_ instance – we just pickle it
        payload = pickle.dumps((self.topic, msg))
        _lowcmd_sock.send(payload)


class ChannelSubscriber:
    # api: __init__, Init, Read
    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type

    def Init(self):
        pass

    def Read(self, timeout_ms=None):
        """Block until we get a lowstate, optionally with timeout (ms)."""
        if timeout_ms is None:
            payload = _lowstate_sock.recv()
        else:
            poller = zmq.Poller()
            poller.register(_lowstate_sock, zmq.POLLIN)
            events = dict(poller.poll(timeout_ms))
            if _lowstate_sock not in events:
                return None
            payload = _lowstate_sock.recv()

        topic, msg = pickle.loads(payload)
        # you can assert topic == self.topic, but not necessary if you only use one
        return msg
