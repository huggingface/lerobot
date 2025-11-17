# XVLA Custom Processor Steps

Three custom processor steps have been implemented for XVLA that encapsulate the preprocessing and postprocessing logic from `lerobot_eval.py`.

## Processor Steps

### 1. XVLAImageScaleProcessorStep
**Registry Name:** `xvla_image_scale`

Scales image observations by 255 (from [0,1] to [0,255] range).

```python
XVLAImageScaleProcessorStep(
    image_keys=None  # Auto-detects "observation.images.*" or specify list
)
```

### 2. XVLAAddDomainIdProcessorStep  
**Registry Name:** `xvla_add_domain_id`

Adds `domain_id` tensor to complementary data for multi-domain support.

```python
XVLAAddDomainIdProcessorStep(
    domain_id=3,     # Domain identifier
    device="cuda"    # Tensor device
)
```

### 3. XVLARotation6DToAxisAngleProcessorStep
**Registry Name:** `xvla_rotation_6d_to_axis_angle`

Converts 6D rotation to axis-angle representation:
- **Input:** [eef(3), rotation_6d(6), gripper(1)] = 10D
- **Output:** [eef(3), axis_angle(3), gripper(1)] = 7D

```python
XVLARotation6DToAxisAngleProcessorStep(
    expected_action_dim=10
)
```

## Integration with Config

These steps can be added to your XVLA policy configuration:

### In Preprocessing Pipeline:
```python
from lerobot.policies.xvla.processor_xvla import (
    XVLAImageScaleProcessorStep,
    XVLAAddDomainIdProcessorStep,
)

preprocessor_steps = [
    RenameObservationsProcessorStep(rename_map={}),
    AddBatchDimensionProcessorStep(),
    XVLAImageScaleProcessorStep(),  # Add this
    TokenizerProcessorStep(...),
    DeviceProcessorStep(device="cuda"),
    XVLAAddDomainIdProcessorStep(domain_id=3),  # Add this
    NormalizerProcessorStep(...),
]
```

### In Postprocessing Pipeline:
```python
from lerobot.policies.xvla.processor_xvla import XVLARotation6DToAxisAngleProcessorStep

postprocessor_steps = [
    UnnormalizerProcessorStep(...),
    XVLARotation6DToAxisAngleProcessorStep(),  # Add this
    DeviceProcessorStep(device="cpu"),
]
```

## Usage in Evaluation

Now your evaluation loop simplifies to:

```python
# Before (from lerobot_eval.py lines 165-184)
observation[f"observation.images.image"] = observation[f"observation.images.image"] * 255
observation[f"observation.images.image2"] = observation[f"observation.images.image2"] * 255
observation = add_envs_task(env, observation)
observation = preprocessor(observation)
observation["domain_id"] = torch.tensor([int(3)], dtype=torch.long).to("cuda")

with torch.inference_mode():
    action = policy.select_action(observation).to("cpu").numpy()
target_eef = action[:, :3]
target_axis = Rotate6D_to_AxisAngle(action[:, 3:9])
target_act = action[:, 9:10]
action_numpy = np.concatenate([target_eef, target_axis, target_act], axis=-1)

# After (clean and simple)
observation = add_envs_task(env, observation)  # Add task
observation = preprocessor(observation)  # Scales images + adds domain_id

with torch.inference_mode():
    action = policy.select_action(observation)
action = postprocessor(action)  # Converts rotation + moves to CPU
action_numpy = action.numpy()
```

## Configuration via Registry

All steps are registered and can be loaded from JSON/YAML config:

```json
{
  "preprocessor": {
    "steps": [
      {"name": "xvla_image_scale"},
      {"name": "xvla_add_domain_id", "domain_id": 3, "device": "cuda"}
    ]
  },
  "postprocessor": {
    "steps": [
      {"name": "xvla_rotation_6d_to_axis_angle", "expected_action_dim": 10}
    ]
  }
}
```

## Implementation Reference

See `processor_groot.py` for similar patterns - these XVLA processors follow the same design:
- Registered with `@ProcessorStepRegistry.register()`
- Implement `__call__`, `transform_features`, and `get_config`
- Operate on `EnvTransition` objects
- Properly handle `transition.copy()` to avoid side effects

