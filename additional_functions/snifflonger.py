# snifflonger.py
import csv, time, sys
from sniff import LowCmdSniffer  # uses latest() -> ok, q, dq, kp, kd, tau

IFACE = sys.argv[1] if len(sys.argv) > 1 else "en7"
DT    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01  # seconds

N = 35

def header(prefix):
    return [f"{prefix}{i}" for i in range(N)]

def main():
    sniffer = LowCmdSniffer(iface=IFACE)

    with open("lowcmd_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + header("kp") + header("kd") + header("q") + header("dq") + header("tau"))
        try:
            while True:
                ok, q, dq, kp, kd, tau = sniffer.latest()
                if ok:
                    t = time.time()
                    # ensure lists (in case numpy arrays)
                    w.writerow([t, *kp.tolist(), *kd.tolist(), *q.tolist(), *dq.tolist(), *tau.tolist()])
                    f.flush()
                time.sleep(DT)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
