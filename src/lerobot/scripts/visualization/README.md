# G1 Visualization Tools

This package contains visualization scripts for the Unitree G1 robot.

## Scripts

- **visualize_g1.py** - 2D dashboard with camera feed, joint plots, and hand controls
- **visualize_g1_3d.py** - 3D URDF viewer using Vuer

## Usage

```bash
# 2D Dashboard (http://localhost:5000)
uv run lerobot/src/lerobot/scripts/visualization/visualize_g1.py

# 3D Viewer (http://localhost:8012)
uv run lerobot/src/lerobot/scripts/visualization/visualize_g1_3d.py
```

See [G1_VISUALIZATION_GUIDE.md](./G1_VISUALIZATION_GUIDE.md) for detailed setup instructions.
