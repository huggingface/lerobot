# Temporal Sampling Strategy Visualization

## How `--sample-interval` Works

### Example: 30 fps dataset, `--sample-interval 1.0` (1 second)

```
Timeline (seconds):  0.0      0.5      1.0      1.5      2.0      2.5      3.0
                     │        │        │        │        │        │        │
Frames:              0───15───30───45───60───75───90───105──120──135──150
                     │        │        │        │        │        │        │
                     ▼                 ▼                 ▼                 ▼
Sampled:            YES      NO       YES      NO       YES      NO       YES
                     │                 │                 │                 │
Task Index:         [0]──────────────>[1]──────────────>[2]──────────────>[3]
                     │                 │                 │                 │
VLM Called:         ✓ Gen             ✓ Gen             ✓ Gen             ✓ Gen
                    dialogue          dialogue          dialogue          dialogue
                     │                 │                 │                 │
Frames 0-29    ─────┘                 │                 │                 │
get task 0                             │                 │                 │
                                       │                 │                 │
Frames 30-59  ────────────────────────┘                 │                 │
get task 1                                               │                 │
                                                         │                 │
Frames 60-89  ──────────────────────────────────────────┘                 │
get task 2                                                                 │
                                                                           │
Frames 90-119 ────────────────────────────────────────────────────────────┘
get task 3
```

## Comparison: Different Sampling Intervals

### `--sample-interval 2.0` (every 2 seconds)
```
Timeline:    0.0      1.0      2.0      3.0      4.0      5.0      6.0
             │        │        │        │        │        │        │
Sampled:    YES      NO       YES      NO       YES      NO       YES
             │                 │                 │                 │
Tasks:      [0]───────────────>[1]───────────────>[2]───────────────>[3]
             
VLM Calls:   4 (fewer calls, faster but less granular)
```

### `--sample-interval 1.0` (every 1 second) - **DEFAULT**
```
Timeline:    0.0   0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0   5.5   6.0
             │     │     │     │     │     │     │     │     │     │     │     │     │
Sampled:    YES   NO   YES   NO   YES   NO   YES   NO   YES   NO   YES   NO   YES
             │           │           │           │           │           │           │
Tasks:      [0]─────────>[1]─────────>[2]─────────>[3]─────────>[4]─────────>[5]─────>[6]
             
VLM Calls:   7 (balanced coverage and speed)
```

### `--sample-interval 0.5` (every 0.5 seconds)
```
Timeline:    0.0  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0  4.5  5.0  5.5  6.0
             │    │    │    │    │    │    │    │    │    │    │    │    │
Sampled:    YES  YES  YES  YES  YES  YES  YES  YES  YES  YES  YES  YES  YES
             │    │    │    │    │    │    │    │    │    │    │    │    │
Tasks:      [0]─>[1]─>[2]─>[3]─>[4]─>[5]─>[6]─>[7]─>[8]─>[9]─>[10]>[11]>[12]
             
VLM Calls:   13 (high granularity, slower but more detailed)
```

## Episode Boundaries

The script always samples the **first frame** of each episode:

```
Episode 0                          Episode 1                          Episode 2
├─────────────────────────────────┤├─────────────────────────────────┤├──────...
│                                 ││                                 ││
Frame: 0    30    60    90   120  130   160   190   220  250  260   290  320
Time:  0.0  1.0   2.0   3.0  4.0  0.0   1.0   2.0   3.0  4.0  0.0   1.0  2.0
       │    │     │     │    │    │     │     │     │    │    │     │    │
       ▼    ▼     ▼     ▼    ▼    ▼     ▼     ▼     ▼    ▼    ▼     ▼    ▼
Sample:YES  YES   YES   YES  YES  YES   YES   YES   YES  YES  YES   YES  YES
       │    │     │     │    │    │     │     │     │    │    │     │    │
Task:  0────1─────2─────3────4    5─────6─────7─────8────9    10────11───12

Note: Frames 0, 130, 260 are ALWAYS sampled (episode starts)
      Even if they're within the sample-interval window
```

## Real-World Example: svla_so101_pickplace Dataset

Typical stats:
- **Total episodes**: 50
- **Avg episode length**: 300 frames (10 seconds at 30 fps)
- **Total frames**: 15,000

### Without Sampling (every frame)
```
Frames processed:    15,000
VLM calls:           15,000
Time estimate:       ~5 hours
Unique tasks:        ~12,000 (lots of duplicates)
```

### With `--sample-interval 1.0` (every 1 second)
```
Frames processed:    15,000 ✓
VLM calls:           500
Time estimate:       ~10 minutes
Unique tasks:        ~450 (meaningful variety)
Efficiency gain:     30x faster
```

### With `--sample-interval 2.0` (every 2 seconds)
```
Frames processed:    15,000 ✓
VLM calls:           250
Time estimate:       ~5 minutes
Unique tasks:        ~220
Efficiency gain:     60x faster
```

## Key Points

1. **All frames get labeled**: Every frame gets a `task_index_high_level`
2. **Only sampled frames call VLM**: Huge efficiency gain
3. **Temporal coherence**: Nearby frames share the same task
4. **Episode-aware**: Always samples episode starts
5. **Configurable**: Adjust `--sample-interval` based on your needs

## Choosing Your Sampling Interval

| Use Case | Recommended Interval | Why |
|----------|---------------------|-----|
| Quick testing | 2.0s | Fastest iteration |
| Standard training | 1.0s | Good balance |
| High-quality dataset | 0.5s | Better coverage |
| Fine-grained control | 0.33s | Very detailed |
| Dense annotations | 0.1s | Nearly every frame |

**Rule of thumb**: Match your sampling interval to your typical skill duration.
If skills last 1-3 seconds, sampling every 1 second captures each skill multiple times.

