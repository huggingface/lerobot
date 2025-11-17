# XVLA Custom Processor Steps - Implementation Summary

## Overview
Implemented three custom processor steps for XVLA that encapsulate the preprocessing and postprocessing logic previously scattered in `lerobot_eval.py` (lines 165-184).

## Files Modified

### 1. `/src/lerobot/policies/xvla/processor_xvla.py`
**Changes:**
- Added imports: `dataclass`, `numpy`, `Rotate6D_to_AxisAngle`, processor core types
- Implemented 3 new processor step classes (all registered with `ProcessorStepRegistry`)

**New Classes:**

#### `XVLAImageScaleProcessorStep` 
- **Registry Name:** `xvla_image_scale`
- **Purpose:** Scales image observations by 255 (converts [0,1] to [0,255])
- **Configuration:** 
  - `image_keys: list[str] | None` - Auto-detects or specify image keys
- **Location:** Lines 93-140

#### `XVLAAddDomainIdProcessorStep`
- **Registry Name:** `xvla_add_domain_id`  
- **Purpose:** Adds domain_id tensor to complementary data
- **Configuration:**
  - `domain_id: int = 3` - Domain identifier
  - `device: str = "cuda"` - Tensor device
- **Location:** Lines 143-192

#### `XVLARotation6DToAxisAngleProcessorStep`
- **Registry Name:** `xvla_rotation_6d_to_axis_angle`
- **Purpose:** Converts 6D rotation to axis-angle and reorganizes action dimensions
  - Input: [eef(3), rotation_6d(6), gripper(1)] = 10D
  - Output: [eef(3), axis_angle(3), gripper(1)] = 7D
- **Configuration:**
  - `expected_action_dim: int = 10`
- **Location:** Lines 195-255

### 2. `/src/lerobot/policies/xvla/README_PROCESSORS.md` (NEW)
Comprehensive documentation covering:
- Processor step descriptions and configurations
- Integration examples for preprocessing/postprocessing pipelines
- Before/after comparison showing simplified evaluation code
- JSON/YAML configuration examples
- Reference to Groot processor patterns

## Key Features

### 1. **Registry-Based Architecture**
All processors are registered with `@ProcessorStepRegistry.register()`, enabling:
- Instantiation from configuration files
- Serialization/deserialization with policies
- Easy discovery and debugging

### 2. **Proper ProcessorStep Interface**
Each processor implements:
- `__call__(transition: EnvTransition) -> EnvTransition` - Main processing logic
- `transform_features(features) -> features` - Feature contract declaration
- `get_config() -> dict` - Serializable configuration

### 3. **Safe Data Handling**
- All processors use `transition.copy()` to avoid side effects
- Proper handling of missing/None values
- Device-aware tensor operations

### 4. **Configurable and Reusable**
- All parameters exposed in `get_config()`
- Can be customized per deployment
- Works with any XVLA model configuration

## Usage Impact

### Before (from lerobot_eval.py):
```python
# Lines 166-184 - scattered preprocessing/postprocessing
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
```

### After (with custom processors):
```python
# Clean and simple - processors encapsulate all the logic
observation = add_envs_task(env, observation)
observation = preprocessor(observation)  # Includes image scaling + domain_id

with torch.inference_mode():
    action = policy.select_action(observation)
action = postprocessor(action)  # Includes rotation conversion + device transfer
action_numpy = action.numpy()
```

## Design Patterns Followed

1. **Groot Processor Reference:** Followed same patterns as `processor_groot.py`:
   - Dataclass-based configuration
   - Registry registration
   - State management via `get_config()`
   - Proper transition handling

2. **LeRobot Processor Guidelines:** (from `implement_your_own_processor.mdx`):
   - Safe data handling with `copy()`
   - Clear error messages
   - Device/dtype awareness
   - Feature contract declaration

3. **Pipeline Integration:** 
   - Works seamlessly with `PolicyProcessorPipeline`
   - Automatic dict ↔ EnvTransition conversion
   - Composable with other processor steps

## Benefits

1. **Cleaner Code:** Evaluation loop is now much simpler
2. **Maintainable:** Processing logic is centralized and well-documented
3. **Configurable:** All parameters can be adjusted via config files
4. **Reusable:** Can be used across different XVLA deployments
5. **Testable:** Each processor can be tested independently
6. **Serializable:** Processors save/load with the policy

## Testing Recommendations

1. **Unit Tests:**
   - Test each processor with sample transitions
   - Verify image scaling (multiply by 255)
   - Verify domain_id addition and device placement
   - Verify rotation conversion accuracy

2. **Integration Tests:**
   - Test full preprocessing pipeline
   - Test full postprocessing pipeline
   - Verify evaluation loop still works correctly
   - Test with different domain_ids and devices

3. **Configuration Tests:**
   - Test loading processors from config
   - Test serialization/deserialization
   - Test overrides mechanism

## Next Steps

1. **Update XVLA Policy Factory:** Optionally add these processors to the default pipeline in `make_xvla_pre_post_processors()` or document how to add them via config

2. **Update lerobot_eval.py:** Simplify the evaluation code to use the new processors

3. **Add Configuration Examples:** Create sample config files showing processor integration

4. **Add Tests:** Implement unit and integration tests for the new processors

## Notes

- No changes made to `make_xvla_pre_post_processors()` as requested
- Processors are available but not automatically included (must be added via config)
- All processors follow LeRobot conventions and best practices
- Compatible with existing XVLA model configurations

