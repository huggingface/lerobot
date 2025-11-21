import time, threading, numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd

TOPIC = "rt/lowcmd"
N = 35

class LowCmdSniffer:
    def __init__(self, iface="en7"):
        ChannelFactoryInitialize(0, iface)
        self.sub = ChannelSubscriber(TOPIC, HGLowCmd)
        # Use a queue depth; passing None as handler is fine for polling
        self.sub.Init(None, 20)

        self._lock = threading.Lock()
        self._have = False
        self.q   = np.zeros(N)
        self.dq  = np.zeros(N)
        self.kp  = np.zeros(N)
        self.kd  = np.zeros(N)
        self.tau = np.zeros(N)

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            msg = self.sub.Read()
            if msg is not None:
                with self._lock:
                    for i in range(N):
                        mc = msg.motor_cmd[i]
                        self.q[i]   = getattr(mc, "q", 0.0)
                        self.dq[i]  = getattr(mc, "qd", 0.0)  # named qd in HG
                        self.kp[i]  = getattr(mc, "kp", 0.0)
                        self.kd[i]  = getattr(mc, "kd", 0.0)
                        self.tau[i] = getattr(mc, "tau", 0.0)
                    self._have = True
            time.sleep(0.001)

    def latest(self):
        with self._lock:
            return self._have, self.q.copy(), self.dq.copy(), self.kp.copy(), self.kd.copy(), self.tau.copy()

if __name__ == "__main__":
    sniffer = LowCmdSniffer(iface="en7")
    t0 = time.time()
    while time.time() - t0 < 5:
        ok, q, dq, kp, kd, tau = sniffer.latest()
        if ok:
            print("kp :", kp.round(2))
            print("kd :", kd.round(2))
            print("q  :", q.round(4))
            print("dq :", dq.round(4))
            print("tau:", tau.round(3))
            break
        time.sleep(0.01)
    else:
        print("No LowCmd messages seen on rt/lowcmd in 5s (check topic/type/interface).")
