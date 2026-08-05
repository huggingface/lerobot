# Task1 Real24 + LocalSim gap24 vs full48 Eval48

This experiment freezes the software-only gate for a paired real evaluation of two
fixed ACT 100k checkpoints:

- `real24_localsim24_gap`: Real24 plus the 24 LocalSim poses absent from Real24.
- `real24_localsim48_full`: Real24 plus all 48 LocalSim poses.

The evaluator reuses the existing frozen Eval48 bank without selecting new poses.
Each of the 48 poses is followed by both models, with the first model alternating
by pose order. The run contract remains official-send, no runner clamp or 5-degree
limiter, no catch-up, 20 Hz, full 30-second window, and the same interpolated ready
pose before and after every trial.

Software preparation never opens serial, camera, robot, or torque. Hardware remains
blocked until a later explicit GO. The first future pair is `r3c2`, X=34 cm,
Y=-1 cm, yaw=45 degrees: gap24 first, then full48 without moving the cube.
