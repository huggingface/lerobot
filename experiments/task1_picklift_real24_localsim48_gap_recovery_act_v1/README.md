# Task1 Real24 + corrected LocalSim ACT pair

This experiment trains the two authorized ACT conditions in fixed order:

- C: Real24 + LocalSim24-gap;
- D: Real24 + LocalSim48-full.

Both use the frozen ACT recipe (seed 1000, 500-step smoke, fresh 100k run,
batch 8 with exactly 4 Real + 4 Simulation samples, ImageNet visual
normalization, action chunk 67). The explicit `source_bindings.json` and the
copied postcollection result manifest replace the discarded brittle string
replacement materializer.

The experiment is offline only. It does not start MuJoCo rollouts or access
camera, serial, robot, torque, or 12V hardware.
