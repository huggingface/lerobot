---
library_name: lerobot
tags:
- robotics
- lerobot
- safetensors
pipeline_tag: robotics
---

# RobotProcessor

## Overview

RobotProcessor is a composable, debuggable post-processing pipeline for robot transitions in the LeRobot framework. It orchestrates an ordered collection of small, functional transforms (steps) that are executed left-to-right on each incoming `EnvTransition`.

## Architecture

The RobotProcessor provides a modular architecture for processing robot environment transitions through a sequence of composable steps. Each step is a callable that accepts a full `EnvTransition` tuple and returns a potentially modified tuple of the same structure.

### EnvTransition Structure

An `EnvTransition` is a 7-tuple containing:
1. **observation**: Current state observation
2. **action**: Action taken (can be None)
3. **reward**: Reward received (float or None)
4. **done**: Episode termination flag (bool or None)
5. **truncated**: Episode truncation flag (bool or None)
6. **info**: Additional information dictionary
7. **complementary_data**: Extra data dictionary

## Key Features

- **Composable Pipeline**: Chain multiple processing steps in a specific order
- **State Persistence**: Save and load processor state using SafeTensors format
- **Hugging Face Hub Integration**: Easy sharing and loading via `save_pretrained()` and `from_pretrained()`
- **Debugging Support**: Step-through functionality to inspect intermediate transformations
- **Hook System**: Before/after step hooks for additional processing or monitoring
- **Device Support**: Move tensor states to different devices (CPU/GPU)
- **Performance Profiling**: Built-in profiling to identify bottlenecks

## Installation

```bash
pip install lerobot
```

## Usage

### Basic Example

```python
from lerobot.processor.pipeline import RobotProcessor
from your_steps import ObservationNormalizer, VelocityCalculator

# Create a processor with multiple steps
processor = RobotProcessor(
    steps=[
        ObservationNormalizer(mean=0, std=1),
        VelocityCalculator(window_size=5),
    ],
    name="my_robot_processor",
    seed=42
)

# Process a transition
obs, info = env.reset()
transition = (obs, None, 0.0, False, False, info, {})
processed_transition = processor(transition)

# Extract processed observation
processed_obs = processed_transition[0]
```

### Saving and Loading

```python
# Save locally
processor.save_pretrained("./my_processor")

# Push to Hugging Face Hub
processor.push_to_hub("username/my-robot-processor")

# Load from Hub
loaded_processor = RobotProcessor.from_pretrained("username/my-robot-processor")
```

### Debugging with Step-Through

```python
# Inspect intermediate results
for idx, intermediate_transition in enumerate(processor.step_through(transition)):
    print(f"After step {idx}: {intermediate_transition[0]}")  # Print observation
```

### Using Hooks

```python
# Add monitoring hook
def log_observation(step_idx, transition):
    print(f"Step {step_idx}: obs shape = {transition[0].shape}")
    return None  # Don't modify transition

processor.register_before_step_hook(log_observation)
```

## Creating Custom Steps

To create a custom processor step, implement the `ProcessorStep` protocol:

```python
from lerobot.processor.pipeline import ProcessorStepRegistry, EnvTransition

@ProcessorStepRegistry.register("my_custom_step")
class MyCustomStep:
    def __init__(self, param1=1.0):
        self.param1 = param1
        self.buffer = []

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs, action, reward, done, truncated, info, comp_data = transition
        # Process observation
        processed_obs = obs * self.param1
        return (processed_obs, action, reward, done, truncated, info, comp_data)

    def get_config(self) -> dict:
        return {"param1": self.param1}

    def state_dict(self) -> dict:
        # Return only torch.Tensor state
        return {}

    def load_state_dict(self, state: dict) -> None:
        # Load tensor state
        pass

    def reset(self) -> None:
        # Clear buffers at episode boundaries
        self.buffer.clear()
```

## Advanced Features

### Device Management

```python
# Move all tensor states to GPU
processor = processor.to("cuda")

# Move to specific device
processor = processor.to(torch.device("cuda:1"))
```

### Performance Profiling

```python
# Profile step execution times
profile_results = processor.profile_steps(transition, num_runs=100)
for step_name, time_ms in profile_results.items():
    print(f"{step_name}: {time_ms:.3f} ms")
```

### Processor Slicing

```python
# Get a single step
first_step = processor[0]

# Create a sub-processor with steps 1-3
sub_processor = processor[1:4]
```

## Model Card Specifications

- **Pipeline Tag**: robotics
- **Library**: lerobot
- **Format**: safetensors
- **License**: Apache 2.0

## Limitations

- Steps must maintain the 7-tuple structure of EnvTransition
- All tensor state must be separated from configuration for proper serialization
- Steps are executed sequentially (no parallel processing within a single transition)

## Citation

If you use RobotProcessor in your research, please cite:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascale, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
