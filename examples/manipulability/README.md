# Manipulability / Singularity Indicator Examples

Tools for visualizing and monitoring the SO-101 arm's manipulability in real time.

## Scripts

### `teleop_with_singularity.py`

Two modes:

- **observe**: Torque OFF — move the arm by hand and see the live singularity indicator (σ_min, condition number, OK/WARNING/CRITICAL).
- **teleop**: Direct joint mirroring from leader to follower with a live singularity overlay and emergency stop on erratic jumps.

```bash
# Observe mode (safe — no motor commands)
python examples/manipulability/teleop_with_singularity.py observe \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --urdf=path/to/so101.urdf

# Teleop mode (with e-stop)
python examples/manipulability/teleop_with_singularity.py teleop \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader \
    --urdf=path/to/so101.urdf
```

### `ellipsoid_viz.py`

Real-time 3D visualization of the velocity manipulability ellipsoid. The ellipsoid shape shows which Cartesian directions the gripper can move easily (long axes) vs. which are blocked (short/collapsed axes). Color indicates proximity to singularity.

```bash
python examples/manipulability/ellipsoid_viz.py \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --urdf=path/to/so101.urdf
```

## What the indicator means

| Status | σ_min | Meaning |
|--------|-------|---------|
| ✅ OK | ≥ 0.015 | Well-conditioned — arm can move freely in all directions |
| ⚠️ WARNING | 0.006–0.015 | Near-singular — reduced dexterity in one direction |
| 🔴 CRITICAL | < 0.006 | At singularity — motion blocked in one Cartesian direction |

The primary singularity for the SO-101 occurs when the elbow straightens, making the upper arm and forearm collinear. This typically happens when the arm is fully extended (pointing up or forward).
