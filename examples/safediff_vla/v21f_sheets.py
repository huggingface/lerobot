"""Contact sheets for visual verification: 6 frames per episode (start, 25%, CLOSE, CLOSE+15, 75%, end), 5 episodes per sheet."""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np
R = Path("outputs/eval/safediff_stageplan_v21_final"); O = R / "review_sheets"; O.mkdir(exist_ok=True); T = ["select_poker", "select_mahjong", "select_drink", "select_book", "select_fruit", "select_toy"]
def tiles(t, s):
    d = R / t / f"seed{s}"; r = json.load(open(d / "result.json")); sl = json.load(open(d / "stage_log.json")); tc = next((i for i, x in enumerate(sl) if x["closed_trig"]), None); cap = cv2.VideoCapture(sorted(glob.glob(str(d / "videos/*.mp4")))[0]); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); c = tc if tc is not None else n // 2
    idx = [0, n // 4, c, min(n - 1, c + 15), 3 * n // 4, n - 1]; row = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read(); fr = cv2.resize(fr, (200, 200)) if ok else np.zeros((200, 200, 3), np.uint8); cv2.putText(fr, f"f{i}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1); row.append(fr)
    row = np.concatenate(row, 1); cv2.putText(row, f"{t[7:]} {s} native={r['success']} len={r['len']}", (3, 195), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0) if r["success"] else (0, 0, 255), 1); return row
pos = [(t, s) for t in T for s in range(3000, 3020) if json.load(open(R / t / f"seed{s}" / "result.json"))["success"]]
fail = {t: [s for s in range(3000, 3020) if not json.load(open(R / t / f"seed{s}" / "result.json"))["success"]] for t in T}
sets = {"pos": pos, "failPM": [(t, s) for t in T[:2] for s in fail[t]], "failOther": [(t, s) for t in T[2:] for s in fail[t][:4]]}
json.dump({k: v for k, v in sets.items()}, open(O / "index.json", "w"))
for name, L in sets.items():
    for k in range(0, len(L), 5): cv2.imwrite(str(O / f"{name}_{k // 5:02d}.jpg"), np.concatenate([tiles(t, s) for t, s in L[k:k + 5]], 0))
print({k: len(v) for k, v in sets.items()})
