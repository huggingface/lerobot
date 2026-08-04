# Task1 Real24-only vs Mixed v2 paired real eval24 review

The independent canonical-video review agrees with all 24 immutable operator
success labels. No adjudication artifact is required.

| Model | Reviewed success | Rate | Reviewed failure |
| --- | ---: | ---: | ---: |
| Real24-only | 10/12 | 83.3% | 2 |
| Mixed v2 | 9/12 | 75.0% | 3 |

The paired difference is -1 success for Mixed v2, or -8.3 percentage points.
Paired outcomes are 8 both-success, 1 both-failure, 2 Real24-only-only, and
1 Mixed-v2-only. All five reviewed failures are visible `missed_grasp`
failures: repeated approaches remain horizontally offset and the cube stays
table-supported with no qualifying bilateral unsupported lift.

| Cell | Real24-only | Mixed v2 | Paired outcome |
| --- | --- | --- | --- |
| r1c1 | success | success | both success |
| r1c2 | success | failure | Real24-only only |
| r1c3 | success | success | both success |
| r1c4 | success | success | both success |
| r2c1 | success | success | both success |
| r2c2 | success | failure | Real24-only only |
| r2c3 | failure | success | Mixed v2 only |
| r2c4 | success | success | both success |
| r3c1 | success | success | both success |
| r3c2 | failure | failure | both failure |
| r3c3 | success | success | both success |
| r3c4 | success | success | both success |

All 24 MP4s were verified as canonical 640x480 at 20 FPS. Every video contains
one frame per actual policy tick: 14,232 frames and 14,232 ticks in total.
Every final tick timestamp is within 29.953-29.981 seconds of the full
30-second wall-clock policy window. The no-catch-up loop emitted 593 actual
ticks per trial; the fixed-rate MP4 playback duration is therefore 29.65
seconds without invented frames.

The review first inspected each full-duration video at 2 FPS. The five
independently classified failures were then rechecked across the full duration
at 4 FPS to rule out a qualifying lift hidden between 0.5-second samples.

This is a descriptive result from one fixed paired real evaluation session. It
does not by itself establish a causal training-data effect or a paper-level
performance conclusion.
