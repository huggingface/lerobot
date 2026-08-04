# Task1 paired real eval24 canonical-video review v1

The independent canonical-video review agrees with all 24 immutable operator
labels. No adjudication artifact is required.

| Model | Reviewed success | Rate | Reviewed failure |
| --- | ---: | ---: | ---: |
| Real24-only | 11/12 | 91.7% | 1 |
| Real24 + Quest-Sim24 | 8/12 | 66.7% | 4 |

Paired cell outcomes are 8 both-success, 1 both-failure, 3 Real24-only-only,
and 0 mixed-only. All five reviewed failures are visible missed-grasp failures:
the cube remains table-supported with no qualifying bilateral unsupported lift.

| Cell | Real24-only | Mixed | Paired outcome |
| --- | --- | --- | --- |
| r1c1 | success | success | both success |
| r1c2 | success | success | both success |
| r1c3 | success | failure | Real24-only only |
| r1c4 | success | success | both success |
| r2c1 | success | failure | Real24-only only |
| r2c2 | success | success | both success |
| r2c3 | success | success | both success |
| r2c4 | success | success | both success |
| r3c1 | success | success | both success |
| r3c2 | failure | failure | both failure |
| r3c3 | success | failure | Real24-only only |
| r3c4 | success | success | both success |

All 24 MP4s were verified as 640x480 at 20 FPS. Each video contains one frame
per actual policy tick: 14,234 frames and 14,234 ticks in total. Every final
tick timestamp is within 29.956-29.997 seconds of the 30-second policy window.
The fixed-rate MP4 playback duration is 29.65 or 29.70 seconds because the
no-catch-up loop emitted 593 or 594 actual ticks rather than inventing frames.

The previously committed operator result index reports 14,233 total ticks.
That single index field is stale by one: the immutable operator manifest,
operator summary, per-trial JSONL files, and MP4 frame counts all independently
resolve to 14,234. The old index and all operator labels remain unchanged.

This is a descriptive review of one frozen paired real evaluation. It is not a
causal conclusion about mixed training data or a paper-level effect claim.
