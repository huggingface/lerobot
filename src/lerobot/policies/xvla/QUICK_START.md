# XVLA Custom Processors - Quick Start

## What Was Implemented

Three custom processor steps that simplify XVLA evaluation by encapsulating preprocessing and postprocessing logic:

```
┌─────────────────────────────────────────────────────────────┐
│  PREPROCESSING PIPELINE                                     │
├─────────────────────────────────────────────────────────────┤
│  1. RenameObservationsProcessorStep                         │
│  2. AddBatchDimensionProcessorStep                          │
│  3. XVLAImageScaleProcessorStep          ← NEW              │
│     └─ Scales images by 255                                 │
│  4. TokenizerProcessorStep                                  │
│  5. DeviceProcessorStep                                     │
│  6. XVLAAddDomainIdProcessorStep         ← NEW              │
│     └─ Adds domain_id tensor                                │
│  7. NormalizerProcessorStep                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  POSTPROCESSING PIPELINE                                    │
├─────────────────────────────────────────────────────────────┤
│  1. UnnormalizerProcessorStep                               │
│  2. XVLARotation6DToAxisAngleProcessorStep  ← NEW           │
│     └─ Converts 6D rotation to axis-angle (10D → 7D)        │
│  3. DeviceProcessorStep(device="cpu")                       │
└─────────────────────────────────────────────────────────────┘
```

## Simplest Usage

### Option 1: Import and Use Directly

```python
from lerobot.policies.xvla.processor_xvla import (
    XVLAImageScaleProcessorStep,
    XVLAAddDomainIdProcessorStep,
    XVLARotation6DToAxisAngleProcessorStep,
)

# Add to your existing preprocessor steps
preprocessor = PolicyProcessorPipeline(
    steps=[
        # ... your existing steps ...
        XVLAImageScaleProcessorStep(),
        # ... more steps ...
        XVLAAddDomainIdProcessorStep(domain_id=3),
    ]
)

# Add to your postprocessor steps
postprocessor = PolicyProcessorPipeline(
    steps=[
        XVLARotation6DToAxisAngleProcessorStep(),
        DeviceProcessorStep(device="cpu"),
    ]
)
```

### Option 2: Load from Config

```python
# In your config.json or YAML:
{
    "preprocessor_steps": [
        {"name": "xvla_image_scale"},
        {"name": "xvla_add_domain_id", "domain_id": 3, "device": "cuda"}
    ],
    "postprocessor_steps": [
        {"name": "xvla_rotation_6d_to_axis_angle", "expected_action_dim": 10}
    ]
}

# Then load:
preprocessor = PolicyProcessorPipeline.from_pretrained("path/to/config")
```

## Evaluation Loop Comparison

### ❌ Old Way (Manual Processing)
```python
# Scattered preprocessing
observation["observation.images.image"] *= 255
observation["observation.images.image2"] *= 255
observation = add_envs_task(env, observation)
observation = preprocessor(observation)
observation["domain_id"] = torch.tensor([3], dtype=torch.long).to("cuda")

# Policy inference
action = policy.select_action(observation)

# Manual postprocessing
target_eef = action[:, :3]
target_axis = Rotate6D_to_AxisAngle(action[:, 3:9])
target_act = action[:, 9:10]
action = np.concatenate([target_eef, target_axis, target_act], axis=-1)
```

### ✅ New Way (With Custom Processors)
```python
# All preprocessing in one call
observation = add_envs_task(env, observation)
observation = preprocessor(observation)  # Includes scaling + domain_id

# Policy inference
action = policy.select_action(observation)

# All postprocessing in one call
action = postprocessor(action)  # Includes rotation conversion
```

**Result:** 13 lines → 6 lines of cleaner, more maintainable code!

## Quick Reference

| Processor | Purpose | Config Key | Default |
|-----------|---------|------------|---------|
| **XVLAImageScaleProcessorStep** | Scale images by 255 | `xvla_image_scale` | Auto-detect images |
| **XVLAAddDomainIdProcessorStep** | Add domain_id tensor | `xvla_add_domain_id` | domain_id=3, device="cuda" |
| **XVLARotation6DToAxisAngleProcessorStep** | Convert 6D→axis-angle | `xvla_rotation_6d_to_axis_angle` | expected_action_dim=10 |

## Key Benefits

1. ✅ **Clean code** - No scattered preprocessing logic
2. ✅ **Configurable** - Adjust via config files
3. ✅ **Reusable** - Works across different XVLA setups
4. ✅ **Serializable** - Saves/loads with policy
5. ✅ **Testable** - Each processor can be tested independently
6. ✅ **Registry-based** - Easy instantiation from config

## Next Steps

1. **Update your evaluation script** to use the new processors
2. **Add processors to your config** if using config-based loading
3. **Test with your specific XVLA model** to ensure compatibility
4. **Adjust parameters** as needed (domain_id, device, etc.)

For detailed documentation, see `README_PROCESSORS.md`.

