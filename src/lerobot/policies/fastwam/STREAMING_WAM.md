## Opt-in eager Streaming-WAM

`FastWAMConfig.streaming_wam` defaults to `False`. Enable it only with a converted
Streaming-WAM AC-Stream checkpoint. The eager implementation uses a 32-action
horizon, 16-action stride, 8-action delay, and one consistency step. It supports
one observation at a time and one inference worker. It does not include the
reference CUDA accelerated runtime or staged consistency distillation.

This implementation adapts Streaming-WAM by Xuyao Huang et al., copyright 2026
Shaohua Pan, under Apache-2.0. See `LICENSE.streaming-wam` and
`NOTICE.streaming-wam`. The source revision is
`ff36d6e66514afd65bf957a06a0a2410f182b0e3`; the Hub release is
`8dcc67ce14c72a5cce0c04c87a518b4ccfec217e` in
[`SJTU-DENG-Lab/Streaming-WAM`](https://huggingface.co/SJTU-DENG-Lab/Streaming-WAM).
The paper is [Streaming-WAM](https://arxiv.org/abs/2609.28927).

Download `ac-stream/model.pt` and `ac-stream/dataset_stats.json` from that exact
Hub revision. Convert them locally:

```bash
uv run python -m lerobot.policies.fastwam.convert_streaming_checkpoint \
    --checkpoint /path/ac-stream/model.pt \
    --stats /path/ac-stream/dataset_stats.json \
    --output /path/native-streaming-wam
```

The converter memory-maps the BF16 checkpoint, checks metadata and every key,
shape, and dtype, saves native safetensors and processors, then requires a strict
public policy reload. Use an external RSS and system-memory guard on constrained
hosts. Frozen Wan VAE/tokenizer assets must already be cached for offline use.
The converted config leaves the text encoder unloaded; provide cached `context`
and `context_mask`, or explicitly enable and load the existing text encoder.

Load without allocating a duplicate trainable model:

```python
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies import make_pre_post_processors

policy = FastWAMPolicy.from_pretrained(
    "/path/native-streaming-wam",
    strict=True,
    init_empty_weights=True,
    local_files_only=True,
)
pre, post = make_pre_post_processors(policy.config, pretrained_path="/path/native-streaming-wam")
try:
    # Serial control loop: supply an observation and cached text context each time.
    batch = pre(observation)
    batch.update(context=context, context_mask=context_mask)
    action = post(policy.select_action(batch))
    execute_action(action)
    policy.mark_action_executed()
finally:
    policy.close()
```

Repeat the loop body for each action. A returned action does not acknowledge
execution. Call `mark_action_executed()` exactly once after execution completes.
Inference launches after eight acknowledged actions. At each stride boundary,
the loop waits for a late continuation. Exceptions propagate to the caller.
`reset()` and `drop_queued_actions()` drain pending work and discard the old
prefix. `close()` drains work and releases the thread; the closed policy cannot
select more actions. Call these methods from the control-loop thread.

The worker copies observations at launch and retains normalized actions. Apply
the saved postprocessor once to each returned action. The release uses minmax
state/action statistics, a normalized state clamp of `[-5, 5]`, and a strict
`> 0.5` gripper threshold after denormalization. Its FastWAM-format LIBERO evaluator
does not clamp normalized actions before denormalization.

Supply float32, byte-derived RGB in `[0, 1]`, with cameras already resized to
224 by 224 and ordered as `observation.images.image`, then
`observation.images.wrist_image`. The concatenated image is 224 by 448. This path
preserves the release's cast-before-scaling pixel convention. Camera orientation
belongs to the caller; the reference LIBERO caller rotates raw cameras by 180
degrees before PIL bilinear resizing. Streaming mode rejects implicit resizing.

The tiny CPU optimizer/save/reload test checks the existing public flow-loss API.
It does not reproduce the release's staged training objective. The optional
manual proof in `tests/policies/fastwam/streaming_proof.py` compares the original
pinned eager definitions with shared cached latent/text tensors. Its CPU timing
includes a current synchronous FastWAM control and a synchronous Streaming-WAM
control with the same weights. Those checks do not establish encoder parity,
CUDA deployment latency, LIBERO success, or real-arm quality.
