# Implement your own Robot Processor

In this tutorial, you'll learn how to implement your own Robot Processor.
It begins by exploring the need for a custom processor, then uses the `NormalizerProcessorStep` as the running example to explain how to implement, configure, and serialize a processor. Finally, it lists all helper processors that ship with LeRobot.

## Why would you need a custom processor?

In most cases, when reading raw data from sensors or when models output actions, you need to process this data to make it compatible with your target system. For example, a common need is normalizing data ranges to make them suitable for neural networks.

LeRobot's `NormalizerProcessorStep` handles this crucial task:

```python
# Input: raw joint positions in [0, 180] degrees
raw_action = torch.tensor([90.0, 45.0, 135.0])

# After processing: normalized to [-1, 1] range for model training
normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=dataset_stats)
normalized_result = normalizer(transition)
# ...
```

Other common processing needs include:

- **Device placement**: Moving tensors between CPU/GPU and converting data types
- **Format conversion**: Transforming between different data structures
- **Batching**: Adding/removing batch dimensions for model compatibility
- **Safety constraints**: Applying limits to robot commands

```python
# Example pipeline combining multiple processors
pipeline = PolicyProcessorPipeline([
    RenameObservationsProcessorStep(rename_map={}),
    AddBatchDimensionProcessorStep(),
    NormalizerProcessorStep(features=features, stats=stats),
    DeviceProcessorStep(device="cuda"),
    # ...
])
```

LeRobot provides a pipeline mechanism to implement sequences of processing steps for both input data and output actions, making it easy to compose these transformations in the right order for optimal performance.

## How to implement your own processor?

We'll use the `NormalizerProcessorStep` as our main example because it demonstrates essential processor patterns including state management, configuration serialization, and tensor handling that you'll commonly need.

Prepare the sequence of processing steps necessary for your problem. A processor step is a class that implements the following methods:

- `__call__`: implements the processing step for the input transition.
- `get_config`: gets the configuration of the processor step.
- `state_dict`: gets the state of the processor step.
- `load_state_dict`: loads the state of the processor step.
- `reset`: resets the state of the processor step.
- `feature_contract`: displays the modification to the feature space during the processor step.

### Implement the `__call__` method

The `__call__` method is the core of your processor step. It takes an `EnvTransition` and returns a modified `EnvTransition`. Here's how the `NormalizerProcessorStep` works:

```python
@dataclass
@ProcessorStepRegistry.register("normalizer_processor")
class NormalizerProcessorStep(ProcessorStep):
    """Normalize observations/actions using dataset statistics."""

    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    eps: float = 1e-8
    _tensor_stats: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Convert stats to tensors for efficient computation."""
        self.stats = self.stats or {}
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=torch.float32)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        # Normalize observations
        # ...
        # Normalize action
        # ...
        return new_transition

```

See the full implementation in `src/lerobot/processor/normalize_processor.py` for complete details.

**Key principles:**

- **Always use `transition.copy()`** to avoid side effects
- **Handle both observations and actions** consistently
- **Separate config from state**: `get_config()` returns JSON-serializable params, `state_dict()` returns tensors
- **Convert stats to tensors** in `__post_init__()` for efficient computation

### Configuration and State Management

Processors support serialization through three methods that separate configuration from tensor state. The `NormalizerProcessorStep` demonstrates this perfectly - it carries dataset statistics (tensors) in its state, and hyperparameters in its config:

```python
# Continuing the NormalizerProcessorStep example...

def get_config(self) -> dict[str, Any]:
    """JSON-serializable configuration (no tensors)."""
    return {
        "eps": self.eps,
        "features": {k: {"type": v.type.value, "shape": v.shape} for k, v in self.features.items()},
        "norm_map": {ft.value: nm.value for ft, nm in self.norm_map.items()},
        # ...
    }

def state_dict(self) -> dict[str, torch.Tensor]:
    """Tensor state only (e.g., dataset statistics)."""
    flat: dict[str, torch.Tensor] = {}
    for key, sub in self._tensor_stats.items():
        for stat_name, tensor in sub.items():
            flat[f"{key}.{stat_name}"] = tensor.cpu()  # Always save to CPU
    return flat

def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    """Restore tensor state at runtime."""
    self._tensor_stats.clear()
    for flat_key, tensor in state.items():
        key, stat_name = flat_key.rsplit(".", 1)
        # Load to processor's configured device
        self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
            dtype=torch.float32, device=self.device
        )
        # ...
```

**Usage:**

```python
# Save (e.g., inside a policy)
config = normalizer.get_config()
tensors = normalizer.state_dict()

# Restore (e.g., loading a pretrained policy)
new_normalizer = NormalizerProcessorStep(**config)
new_normalizer.load_state_dict(tensors)
# Now new_normalizer has the same stats and configuration
```

### Transform features

The `transform_features` method defines how your processor transforms feature names and shapes. This is crucial for policy configuration and debugging.

For `NormalizerProcessorStep`, features are typically preserved unchanged since normalization doesn't alter keys or shapes:

```python
def transform_features(self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
    """Normalization preserves all feature definitions."""
    return features  # No changes to feature structure
    # ...
```

When your processor renames or reshapes data, implement this method to reflect the mapping for downstream components. For example, a simple rename processor:

```python
def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
    # Simple renaming
    if "pixels" in features:
        features["observation.image"] = features.pop("pixels")

    # Pattern-based renaming
    for key in list(features.keys()):
        if key.startswith("env_state."):
            suffix = key[len("env_state."):]
            features[f"observation.{suffix}"] = features.pop(key)
            # ...

    return features
```

**Key principles:**

- Use `features.pop(old_key)` to remove and get the old feature
- Use `features[new_key] = old_feature` to add the renamed feature
- Always return the modified features dictionary
- Document transformations clearly in the docstring

### Using overrides

You can override step parameters at load-time using `overrides`. This is handy for non-serializable objects or site-specific settings. It works both in policy factories and with `DataProcessorPipeline.from_pretrained(...)`.

**Foundational model adaptation**: This is particularly useful when working with foundational pretrained policies where you rarely have access to the original training statistics. You can inject your own dataset statistics to adapt the normalizer to your specific robot or environment data.

Example: during policy evaluation on the robot, override the device and rename map.
Use this to run a policy trained on CUDA on a CPU-only robot, or to remap camera keys when the robot uses different names than the dataset.

Direct usage with `from_pretrained`:

```python
from lerobot.processor import RobotProcessorPipeline

# Load a foundational policy trained on diverse robot data
# but adapt normalization to your specific robot/environment
new_stats = LeRobotDataset(repo_id="username/my-dataset").meta.stats
processor = RobotProcessorPipeline.from_pretrained(
    "huggingface/foundational-robot-policy",  # Pretrained foundation model
    overrides={
        "normalizer_processor": {"stats": new_stats},     # Inject your robot's statistics
        "device_processor": {"device": "cuda:0"},         # registry name for registered steps
        "rename_processor": {"rename_map": robot_key_map}, # Map your robot's observation keys
        # ...
    },
)
```

## Best Practices

Based on analysis of all LeRobot processor implementations, here are the key patterns and practices:

### 1. **Safe Data Handling**

Always create copies of input data to avoid unintended side effects. Use `transition.copy()` and `observation.copy()` rather than modifying data in-place. This prevents your processor from accidentally affecting other components in the pipeline.

Check for required data before processing and handle missing data gracefully. If your processor expects certain keys (like `"pixels"` for image processing), validate their presence first. For optional data, use safe access patterns like `transition.get()` and handle `None` values appropriately.

When data validation fails, provide clear, actionable error messages that help users understand what went wrong and how to fix it.

### 2. **Choose Appropriate Base Classes**

LeRobot provides specialized base classes that reduce boilerplate code and ensure consistency. Use `ObservationProcessorStep` when you only need to modify observations, `ActionProcessorStep` for action-only processing, and `RobotActionProcessorStep` specifically for dictionary-based robot actions.

Only inherit directly from `ProcessorStep` when you need full control over the entire transition or when processing multiple transition components simultaneously. The specialized base classes handle the transition management for you and provide type safety.

### 3. **Registration and Naming**

Register your processors with descriptive, namespaced names using `@ProcessorStepRegistry.register()`. Use organization prefixes like `"robotics_lab/safety_clipper"` or `"acme_corp/vision_enhancer"` to avoid naming conflicts. Avoid generic names like `"processor"` or `"step"` that could clash with other implementations.

Good registration makes your processors discoverable and enables clean serialization/deserialization when saving and loading pipelines.

### 4. **State Management Patterns**

Distinguish between configuration parameters (JSON-serializable values) and internal state (tensors, buffers). Use dataclass fields with `init=False, repr=False` for internal state that shouldn't appear in the constructor or string representation.

Implement the `reset()` method to clear internal state between episodes. This is crucial for stateful processors that accumulate data over time, like moving averages or temporal filters.

Remember that `get_config()` should only return JSON-serializable configuration, while `state_dict()` handles tensor state separately.

### 5. **Input Validation and Error Handling**

Validate input types and shapes before processing. Check tensor properties like `dtype` and dimensions to ensure compatibility with your algorithms. For robot actions, verify that required pose components or joint values are present and within expected ranges.

Use early returns for edge cases where no processing is needed. Provide clear, descriptive error messages that include the expected vs. actual data types or shapes. This makes debugging much easier for users.

### 6. **Device and Dtype Awareness**

Design your processors to automatically adapt to the device and dtype of input tensors. Internal tensors (like normalization statistics) should match the input tensor's device and dtype to ensure compatibility with multi-GPU training, mixed precision, and distributed setups.

Implement a `to()` method that moves your processor's internal state to the specified device. Check device/dtype compatibility at runtime and automatically migrate internal state when needed. This pattern enables seamless operation across different hardware configurations without manual intervention.

## Conclusion

You now have all the tools to implement custom processors in LeRobot! The key steps are:

1. **Define your processor** as a dataclass with the required methods (`__call__`, `get_config`, `state_dict`, `load_state_dict`, `reset`, `transform_features`)
2. **Register it** using `@ProcessorStepRegistry.register("name")` for discoverability
3. **Integrate it** into a `DataProcessorPipeline` with other processing steps
4. **Use base classes** like `ObservationProcessorStep` when possible to reduce boilerplate
5. **Implement device/dtype awareness** to support multi-GPU and mixed precision setups

The processor system is designed to be modular and composable, allowing you to build complex data processing pipelines from simple, focused components. Whether you're preprocessing sensor data for training or post-processing model outputs for robot execution, custom processors give you the flexibility to handle any data transformation your robotics application requires.

Key principles for robust processors:

- **Device/dtype adaptation**: Internal tensors should match input tensors
- **Clear error messages**: Help users understand what went wrong
- **Base class usage**: Leverage specialized base classes to reduce boilerplate
- **Feature contracts**: Declare data structure changes with `transform_features()`

Start simple, test thoroughly, and ensure your processors work seamlessly across different hardware configurations!
