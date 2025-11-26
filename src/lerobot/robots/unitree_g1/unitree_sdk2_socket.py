import pickle
import zmq

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

_ctx = None
_lowcmd_sock = None
_lowstate_sock = None

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001


def ChannelFactoryInitialize(*args, **kwargs):#DDS to socket bridge
    global _ctx, _lowcmd_sock, _lowstate_sock\
    
    # read socket config
    config = UnitreeG1Config()
    robot_ip = config.robot_ip
    
    _ctx = zmq.Context.instance()

    # lowcmd: robot action
    _lowcmd_sock = _ctx.socket(zmq.PUSH)
    _lowcmd_sock.setsockopt(zmq.CONFLATE, 1)#keep only last message
    _lowcmd_sock.connect(f"tcp://{robot_ip}:{LOWCMD_PORT}")

    # lowstate: robot observation
    _lowstate_sock = _ctx.socket(zmq.SUB)
    _lowstate_sock.setsockopt(zmq.CONFLATE, 1)  # keep only last message
    _lowstate_sock.connect(f"tcp://{robot_ip}:{LOWSTATE_PORT}")
    _lowstate_sock.setsockopt_string(zmq.SUBSCRIBE, "")


class ChannelPublisher: #send action to robot
    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type

    def Init(self):
        pass

    def Write(self, msg):
        _lowcmd_sock.send(pickle.dumps((self.topic, msg)))


class ChannelSubscriber: #read observation from robot
    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type

    def Init(self):
        pass

    def Read(self):
        topic, msg = pickle.loads(_lowstate_sock.recv())
        return msg
