# XVLA Configuration and Evaluation Updates - Summary

## Overview
Updated XVLA configuration files and evaluation script to use the new custom processor steps, eliminating manual preprocessing and postprocessing code.

## Files Modified

### 1. `/src/lerobot/policies/xvla/policy_preprocessor.json`

**Added two new processor steps:**

#### Step 3: `xvla_image_scale` (NEW - Line 14-19)
```json
{
  "registry_name": "xvla_image_scale",
  "config": {
    "image_keys": null
  }
}
```
- **Position:** After `to_batch_processor`, before `tokenizer_processor`
- **Purpose:** Scales images by 255 (converts from [0,1] to [0,255])
- **Replaces:** Manual code `observation["observation.images.image"] *= 255`

#### Step 6: `xvla_add_domain_id` (NEW - Line 38-44)
```json
{
  "registry_name": "xvla_add_domain_id",
  "config": {
    "domain_id": 3,
    "device": "cuda"
  }
}
```
- **Position:** After `device_processor`, before `normalizer_processor`
- **Purpose:** Adds domain_id tensor to complementary data
- **Replaces:** Manual code `observation["domain_id"] = torch.tensor([int(3)], dtype=torch.long).to("cuda")`

**Final preprocessing pipeline order:**
1. `rename_observations_processor`
2. `to_batch_processor`
3. `xvla_image_scale` ⭐ NEW
4. `tokenizer_processor`
5. `device_processor`
6. `xvla_add_domain_id` ⭐ NEW
7. `normalizer_processor`

### 2. `/src/lerobot/policies/xvla/policy_postprocessor.json`

**Added one new processor step and updated device:**

#### Step 2: `xvla_rotation_6d_to_axis_angle` (NEW - Line 23-28)
```json
{
  "registry_name": "xvla_rotation_6d_to_axis_angle",
  "config": {
    "expected_action_dim": 10
  }
}
```
- **Position:** After `unnormalizer_processor`, before `device_processor`
- **Purpose:** Converts 6D rotation to axis-angle (10D → 7D action)
- **Replaces:** Manual code:
  ```python
  target_eef = action[:, :3]
  target_axis = Rotate6D_to_AxisAngle(action[:, 3:9])
  target_act = action[:, 9:10]
  action = np.concatenate([target_eef, target_axis, target_act], axis=-1)
  ```

#### Step 3: `device_processor` (UPDATED - Line 29-35)
- **Changed device:** `"cuda"` → `"cpu"`
- **Purpose:** Move tensors to CPU for environment interaction
- **Replaces:** Manual code `.to("cpu")`

**Final postprocessing pipeline order:**
1. `unnormalizer_processor`
2. `xvla_rotation_6d_to_axis_angle` ⭐ NEW
3. `device_processor` (device changed to "cpu") 🔧 UPDATED

### 3. `/src/lerobot/scripts/lerobot_eval.py`

**Removed manual preprocessing/postprocessing code:**

#### Lines 91-92: Removed import (DELETED)
```python
# REMOVED:
from lerobot.policies.xvla.utils import Rotate6D_to_AxisAngle
```

#### Lines 165-184: Simplified evaluation logic (REPLACED)

**Before (18 lines with manual processing):**
```python
observation[f"observation.images.image"] = observation[f"observation.images.image"] * 255
observation[f"observation.images.image2"] = observation[f"observation.images.image2"] * 255
observation = add_envs_task(env, observation)
observation = preprocessor(observation)
observation["domain_id"] = torch.tensor([int(3)], dtype=torch.long).to("cuda")

with torch.inference_mode():
    action = policy.select_action(observation).to("cpu").numpy()
# action = postprocessor(action)  # THIS WAS COMMENTED OUT
target_eef = action[:, :3]
target_axis = Rotate6D_to_AxisAngle(action[:, 3:9])
target_act = action[:, 9:10]
action_numpy = np.concatenate([target_eef, target_axis, target_act], axis=-1)

# Convert to CPU / numpy.
# action_numpy: np.ndarray = action.to("cpu").numpy()
assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
```

**After (11 lines, clean and simple):**
```python
observation = add_envs_task(env, observation)

# Preprocess observation (includes image scaling and domain_id addition)
observation = preprocessor(observation)

# Policy inference
with torch.inference_mode():
    action = policy.select_action(observation)

# Postprocess action (includes rotation conversion and device transfer to CPU)
action = postprocessor(action)

# Convert to numpy
action_numpy: np.ndarray = action.numpy()
assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
```

## Impact Summary

### Code Reduction
- **Lines removed:** ~13 lines of manual processing code
- **Lines added:** ~7 lines of clean processor calls
- **Net reduction:** ~6 lines + cleaner structure
- **Removed import:** No longer need `Rotate6D_to_AxisAngle` import

### Benefits

1. **✅ Cleaner Code**
   - Evaluation loop is now much simpler and more readable
   - No scattered preprocessing logic
   - Clear separation of concerns

2. **✅ Configuration-Driven**
   - All preprocessing/postprocessing controlled via JSON config
   - Easy to adjust parameters (domain_id, device, etc.) without code changes
   - Can load different configs for different deployments

3. **✅ Maintainable**
   - Processing logic centralized in processor classes
   - Single source of truth for transformations
   - Easier to debug and test

4. **✅ Reusable**
   - Processors work across all XVLA evaluations
   - Can be shared between training and inference
   - Can be serialized with the model

5. **✅ Consistent**
   - Same processing pipeline guaranteed in all contexts
   - No risk of forgetting manual steps
   - Automatic handling of edge cases

## Testing Checklist

Before deploying, verify:

- [ ] Images are scaled correctly (0-255 range)
- [ ] domain_id is added to complementary data
- [ ] 6D rotation correctly converts to axis-angle
- [ ] Actions are 7D after postprocessing
- [ ] Evaluation success rates match previous results
- [ ] Video rendering still works
- [ ] Multi-environment batching works correctly

## Configuration Notes

### Customizing Domain ID
To change the domain ID for different embodiments, edit `policy_preprocessor.json`:
```json
{
  "registry_name": "xvla_add_domain_id",
  "config": {
    "domain_id": 5,  // Change this value
    "device": "cuda"
  }
}
```

### Customizing Image Keys
To scale specific images only, edit `policy_preprocessor.json`:
```json
{
  "registry_name": "xvla_image_scale",
  "config": {
    "image_keys": ["observation.images.image", "observation.images.wrist_cam"]
  }
}
```

### Customizing Action Dimensions
To support different action dimensions, edit `policy_postprocessor.json`:
```json
{
  "registry_name": "xvla_rotation_6d_to_axis_angle",
  "config": {
    "expected_action_dim": 12  // Adjust based on your model
  }
}
```

## Migration Guide

If you have existing XVLA checkpoints without these configs:

1. **Copy the updated JSON files** to your checkpoint directory
2. **No model retraining needed** - processors are data transforms only
3. **Test evaluation** to ensure consistent results
4. **Update any custom evaluation scripts** to use processors

## Related Files

- Custom processors implementation: `/src/lerobot/policies/xvla/processor_xvla.py`
- Documentation: `/src/lerobot/policies/xvla/README_PROCESSORS.md`
- Quick start: `/src/lerobot/policies/xvla/QUICK_START.md`

## Questions?

See the processor documentation in `/src/lerobot/policies/xvla/README_PROCESSORS.md` for detailed usage examples and troubleshooting.

