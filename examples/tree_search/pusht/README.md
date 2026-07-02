# Beam Search For PushT

This repository contains the PushT tree-search experiments and visualizations using Beam Search. It is intended to be used with a compatible LeRobot checkout or fork.
The plotting and animation scripts can inspect saved outputs independently, but
the evaluators rely on LeRobot policy loading, preprocessing, environment
construction, and PushT simulation utilities.

<table>
  <tr>
    <th>Algorithm</th>
    <th>Demo</th>
  </tr>
  <tr>
    <td><img src="docs/assets/search-animation.gif" alt="Search algorithm animation" width="100%"></td>
    <td><img src="docs/assets/search-example-1.gif" alt="PushT search demo" width="100%"></td>
  </tr>
</table>

<p align="center">
  <img src="docs/assets/pusht-frames.png" alt="PushT rollout comparison" width="90%">
</p>

<div class="table-container">
  <table class="table is-bordered is-striped is-hoverable is-fullwidth results-table">
    <caption>
      PushT search performance across Beam Search tree configurations.
    </caption>
    <thead>
      <tr>
        <th>Method</th>
        <th>D</th>
        <th>C</th>
        <th>BW</th>
        <th>Avg. Sum Reward</th>
        <th>Avg. Max Reward</th>
        <th>Success (%)</th>
        <th>ASR (%)</th>
        <th>Runtime (s)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Beam Search</td>
        <td>2</td>
        <td>4</td>
        <td>2</td>
        <td>90.9</td>
        <td>0.77</td>
        <td>18</td>
        <td>56.4</td>
        <td>14</td>
      </tr>
      <tr>
        <td>Beam Search</td>
        <td>4</td>
        <td>2</td>
        <td>2</td>
        <td>90.8</td>
        <td>0.78</td>
        <td>16</td>
        <td>37.5</td>
        <td>10</td>
      </tr>
      <tr>
        <td>Beam Search</td>
        <td>4</td>
        <td>4</td>
        <td>2</td>
        <td>90.9</td>
        <td>0.77</td>
        <td>18</td>
        <td>58.23</td>
        <td>10</td>
      </tr>
      <tr>
        <td>Beam Search</td>
        <td>4</td>
        <td>8</td>
        <td>2</td>
        <td>94.7</td>
        <td>0.77</td>
        <td>18</td>
        <td>75.6</td>
        <td>33</td>
      </tr>
      <tr>
        <td>Beam Search</td>
        <td>4</td>
        <td>8</td>
        <td>4</td>
        <td>103.2</td>
        <td>0.81</td>
        <td>18</td>
        <td>75</td>
        <td>47</td>
      </tr>
      <tr>
        <td>Beam Search</td>
        <td>4</td>
        <td>8</td>
        <td>8</td>
        <td>92.2</td>
        <td>0.80</td>
        <td>10</td>
        <td>78.5</td>
        <td>87</td>
      </tr>
      <tr>
        <td>Beam Search</td>
        <td>8</td>
        <td>4</td>
        <td>2</td>
        <td>90.9</td>
        <td>0.77</td>
        <td>18</td>
        <td>58.23</td>
        <td>16</td>
      </tr>
      <tr>
        <td>ACT (Baseline)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>39.44</td>
        <td>0.36</td>
        <td>0.0</td>
        <td>-</td>
        <td>2</td>
      </tr>
    </tbody>
  </table>
</div>

## Reproduction

Run PushT tree search in parallel:

```bash
python examples/tree_search/pusht/parallel_search_eval.py \
  --policy.path=aadarshram/act_pusht \
  --policy.device=cuda \
  --policy.use_amp=false \
  --episodes=50 \
  --episode-workers=10 \
  --noise-std=40 \
  --depth=3 \
  --num-candidates=3 \
  --beam-width=2 \
  --chunk-size=100 \
  --seed=0 \
  --max_steps=300 \
  --execute-steps=10 \
  --log-every-steps=10 \
  \
  --render-videos=30 \
  --dump_frames=true \
  --plot_policy_trace \
  --dump_search_images=true \
  --video_overlay=false \
  \
  --output_dir="out/50_eps_parallel_viz/n40_d3_c3_b2_e10"
```

Run the ACT PushT baseline with LeRobot:

```bash
lerobot-eval \
  --policy.path=aadarshram/act_pusht \
  --env.type=pusht \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=0 \
  --policy.use_amp=false \
  --policy.device=cuda
```

Detailed script options, visualization tools, and output layouts are documented
in [pusht/README.md](pusht/README.md).
