"""G6 temporal referent stabilisation: ONE state machine (inference-only, no learned parts, no GT) shared by the offline replay and the closed-loop run.
What is locked is the SPATIAL REFERENT (front-image location L of the card the fine matcher selected), never the anchor xyz: the anchor is re-computed at every replan from the CURRENT frame, with the candidate-level fine probability forced onto the candidate that is spatially
associated with L.  Fixed constants (pre-specified, no sweeps): LOCK_P=0.7, MARGIN=0.15, NEED=2 consecutive replans, MAX_MISSING=2 (1 grace replan), association radius = half crop side (72 px, the same value as the proposal dedup radius).
cands: list of dict(box=(x0,y0,side), p=float, fine=(u,v)|None, map_ok=bool) in front-image pixels (480x480)."""
from __future__ import annotations
import numpy as np

LOCK_P, MARGIN, NEED, MAX_MISSING, RADIUS, L_FOLLOW = 0.7, 0.15, 2, 2, 72.0, 60.0


def center(c): x0, y0, s = c["box"]; return np.array([x0 + s / 2, y0 + s / 2], float)


class Stabilizer:
    def __init__(self): self.reset()

    def reset(self): self.locked = False; self.L = None; self.pending = None; self.missing = 0; self.n_switch = 0

    def _assoc(self, cands):
        best = None
        for j, c in enumerate(cands):
            d = float(np.linalg.norm(center(c) - self.L))
            if d <= RADIUS and (best is None or d < best[1]): best = (j, d)
        return None if best is None else best[0]

    def peek(self, cands):
        """read-only association (used for per-frame monitor calls between replans): index of the candidate associated with the current lock, or None."""
        return self._assoc(cands) if self.locked else None

    def step(self, cands) -> dict:
        """called ONCE per planner replan. returns {action: raw|lock|hold|switch|grace|unlock, idx: forced candidate or None}."""
        if not self.locked:
            j = int(np.argmax([c["p"] for c in cands]))
            if cands[j]["p"] >= LOCK_P and cands[j]["map_ok"]:
                self.locked = True; self.L = np.array(cands[j]["fine"] if cands[j]["fine"] is not None else center(cands[j]), float); self.pending = None; self.missing = 0; return {"action": "lock", "idx": j}
            return {"action": "raw", "idx": None}
        a = self._assoc(cands)
        if a is None:
            self.missing += 1
            if self.missing >= MAX_MISSING: self.reset(); return {"action": "unlock", "idx": None}
            return {"action": "grace", "idx": None}
        self.missing = 0; pa = cands[a]["p"]; others = [j for j in range(len(cands)) if j != a and np.linalg.norm(center(cands[j]) - self.L) > RADIUS]
        if others:
            b = max(others, key=lambda j: cands[j]["p"])
            if cands[b]["p"] > pa + MARGIN:
                cb = center(cands[b]); self.pending = {"c": cb, "n": self.pending["n"] + 1} if (self.pending and np.linalg.norm(self.pending["c"] - cb) <= RADIUS) else {"c": cb, "n": 1}
                if self.pending["n"] >= NEED:
                    self.L = np.array(cands[b]["fine"] if cands[b]["fine"] is not None else cb, float); self.pending = None; self.n_switch += 1; return {"action": "switch", "idx": b}
                f = cands[a]["fine"]
                if f is not None and np.linalg.norm(np.asarray(f) - self.L) <= L_FOLLOW: self.L = np.asarray(f, float)
                return {"action": "hold", "idx": a}
        self.pending = None; f = cands[a]["fine"]
        if f is not None and np.linalg.norm(np.asarray(f) - self.L) <= L_FOLLOW: self.L = np.asarray(f, float)
        return {"action": "hold", "idx": a}
