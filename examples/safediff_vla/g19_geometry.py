"""G19 NON-LEARNED geometric action chunk (translation only, no parameters, no sweep).
Pre-specified: h = 8 cm above the target; polyline p0 -> p_above -> g; constant-speed linear interpolation, phase A = 30 steps (chunk index 0..29 ends AT p_above), phase B = 20 steps (index 30..49 ends AT g). First action = p0 + (p_above-p0)/30 (continuous with the current EE). Orientation: not produced (translation is the only thing evaluated; closed loop would keep the current orientation). Gripper: OPEN until EE-target 3D distance <= 3 cm (the existing reactive-close criterion), never from a learned head."""
import numpy as np

H_ABOVE, NA, NB = 0.08, 30, 20


def geo_chunk(ee, g, h=H_ABOVE):
    ee = np.asarray(ee, float); g = np.asarray(g, float); above = g + np.array([0.0, 0.0, h]); a = (np.arange(1, NA + 1) / NA)[:, None]; b = (np.arange(1, NB + 1) / NB)[:, None]
    return np.concatenate([ee + a * (above - ee), above + b * (g - above)], 0)
