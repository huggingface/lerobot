# SO-101 MuJoCo assets

Vendored from `TheRobotStudio/SO-ARM100`, directory `Simulation/SO101`, commit
`fda892cba81032c46c40976a48c9ceadbf40a9ca`.

Only the two small MJCF XML files are included. The referenced STL files are byte-identical
to the assets already fetched by LeRobot's `lerobot/robot-urdfs/so101` cache, so the adapter
reuses that cache instead of adding roughly 16 MB of duplicate meshes to the git repository.
The upstream project is Apache-2.0 licensed.

The jaw endpoints were visually checked with MuJoCo rendering on 2026-07-17: actuator
`-0.17453 rad` is closed and `1.74533 rad` is open. The adapter maps those to LeRobot 0 and
100 respectively.
