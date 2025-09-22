# Debug Your Processor Pipeline

Processor pipelines can be complex, especially when chaining multiple transformation steps.
Unlike simple function calls, pipelines lack natural observability, you can't easily see what happens
between each step or where things go wrong.
This guide provides debugging tools and techniques specifically designed to address these challenges
and help you understand data flow through your pipelines.

We'll explore three complementary debugging approaches: **hooks** for runtime monitoring, **step-through debugging** for detailed inspection, and **feature validation** for catching structural mismatches. Each serves a different purpose and together they provide complete visibility into your pipeline's behavior.

## Understanding Hooks

Hooks are functions that get called at specific points during pipeline execution.
They provide a way to inspect, monitor, or modify data without changing your pipeline code.
Think of them as "event listeners" for your pipeline.

### What is a Hook?

A hook is a callback function that gets automatically invoked at specific moments during pipeline execution.
The concept comes from event-driven programming, imagine you could "hook into" the pipeline's execution flow to observe or react to what's happening.

Think of hooks like inserting checkpoints into your pipeline. Every time the pipeline reaches one of these checkpoints, it pauses briefly to call your hook function, giving you a chance to inspect the current state, log information, and validate data.

A hook is simply a function that accepts two parameters:

- `step_idx: int` - The index of the current processing step (0, 1, 2, etc.)
- `transition: EnvTransition` - The data transition at that point in the pipeline

The beauty of hooks is their non-invasive nature: you can add monitoring, validation, or debugging logic without changing a single line of your pipeline code. The pipeline remains clean and focused on its core logic, while hooks handle the cross-cutting concerns like logging, monitoring, and debugging.

### Before vs After Hooks

The pipeline supports two types of hooks:

- **Before hooks** (`register_before_step_hook`) - Called before each step executes
- **After hooks** (`register_after_step_hook`) - Called after each step completes

```python
def before_hook(step_idx: int, transition: EnvTransition):
    """Called before step processes the transition."""
    print(f"About to execute step {step_idx}")
    # Useful for: logging, validation, setup

def after_hook(step_idx: int, transition: EnvTransition):
    """Called after step has processed the transition."""
    print(f"Completed step {step_idx}")
    # Useful for: monitoring results, cleanup, debugging

processor.register_before_step_hook(before_hook)
processor.register_after_step_hook(after_hook)
```

### Implementing a NaN Detection Hook

Here's a practical example of a hook that detects NaN values:

```python
def check_nans(step_idx: int, transition: EnvTransition):
    """Check for NaN values in observations."""
    obs = transition.get(TransitionKey.OBSERVATION)
    if obs:
        for key, value in obs.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                print(f"NaN detected in {key} at step {step_idx}")

# Register the hook to run after each step
processor.register_after_step_hook(check_nans)

# Process your data - the hook will be called automatically
output = processor(input_data)

# Remove the hook when done debugging
processor.unregister_after_step_hook(check_nans)
```

### How Hooks Work Internally

Understanding the internal mechanism helps you use hooks more effectively. The pipeline maintains two separate lists: one for before-step hooks and another for after-step hooks. When you register a hook, it's simply appended to the appropriate list.

During execution, the pipeline follows a strict sequence: for each processing step, it first calls all before-hooks in registration order, then executes the actual step transformation, and finally calls all after-hooks in registration order. This creates a predictable, sandwich-like structure around each step.

The key insight is that hooks don't change the core pipeline logic—they're purely additive. The pipeline's `_forward` method orchestrates this dance between hooks and processing steps, ensuring that your debugging or monitoring code runs at exactly the right moments without interfering with the main data flow.

Here's a simplified view of how the pipeline executes hooks:

```python
class DataProcessorPipeline:
    def __init__(self):
        self.steps = [...]
        self.before_step_hooks = []  # List of before hooks
        self.after_step_hooks = []   # List of after hooks

    def _forward(self, transition):
        """Internal method that processes the transition through all steps."""
        for step_idx, processor_step in enumerate(self.steps):
            # 1. Call all BEFORE hooks
            for hook in self.before_step_hooks:
                hook(step_idx, transition)

            # 2. Execute the actual processing step
            transition = processor_step(transition)

            # 3. Call all AFTER hooks
            for hook in self.after_step_hooks:
                hook(step_idx, transition)

        return transition

    def register_before_step_hook(self, hook_fn):
        self.before_step_hooks.append(hook_fn)

    def register_after_step_hook(self, hook_fn):
        self.after_step_hooks.append(hook_fn)
```

### Execution Flow

The execution flow looks like this:

```
Input → Before Hook → Step 0 → After Hook → Before Hook → Step 1 → After Hook → ... → Output
```

For example, with 3 steps and both hook types:

```python
def timing_before(step_idx, transition):
    print(f"⏱️  Starting step {step_idx}")

def validation_after(step_idx, transition):
    print(f"✅ Completed step {step_idx}")

processor.register_before_step_hook(timing_before)
processor.register_after_step_hook(validation_after)

# This will output:
# ⏱️  Starting step 0
# ✅ Completed step 0
# ⏱️  Starting step 1
# ✅ Completed step 1
# ⏱️  Starting step 2
# ✅ Completed step 2
```

### Multiple Hooks

You can register multiple hooks of the same type - they execute in the order registered:

```python
def log_shapes(step_idx: int, transition: EnvTransition):
    obs = transition.get(TransitionKey.OBSERVATION)
    if obs:
        print(f"Step {step_idx} observation shapes:")
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")

processor.register_after_step_hook(check_nans)      # Executes first
processor.register_after_step_hook(log_shapes)     # Executes second

# Both hooks will be called after each step in registration order
output = processor(input_data)
```

While hooks are excellent for monitoring specific issues (like NaN detection) or gathering metrics during normal pipeline execution, sometimes you need to dive deeper. When you want to understand exactly what happens at each step or debug complex transformation logic, step-through debugging provides the detailed inspection you need.

## Step-Through Debugging

Step-through debugging is like having a slow-motion replay for your pipeline. Instead of watching your data get transformed in one quick blur from input to output, you can pause and examine what happens after each individual step.

This approach is particularly valuable when you're trying to understand a complex pipeline, debug unexpected behavior, or verify that each transformation is working as expected. Unlike hooks, which are great for automated monitoring, step-through debugging gives you manual, interactive control over the inspection process.

The `step_through()` method is a generator that yields the transition state after each processing step, allowing you to inspect intermediate results. Think of it as creating a series of snapshots of your data as it flows through the pipeline—each snapshot shows you exactly what your data looks like after one more transformation has been applied.

### How Step-Through Works

The `step_through()` method fundamentally changes how the pipeline executes. Instead of running all steps in sequence and only returning the final result, it transforms the pipeline into an iterator that yields intermediate results.

Here's what happens internally: the method starts by converting your input data into the pipeline's internal transition format, then yields this initial state. Next, it applies the first processing step and yields the result. Then it applies the second step to that result and yields again, and so on. Each `yield` gives you a complete snapshot of the transition at that point.

This generator pattern is powerful because it's lazy—the pipeline only computes the next step when you ask for it. This means you can stop at any point, inspect the current state thoroughly, and decide whether to continue. You're not forced to run the entire pipeline just to debug one problematic step.

Instead of running the entire pipeline and only seeing the final result, `step_through()` pauses after each step and gives you the intermediate transition:

```python
# This creates a generator that yields intermediate states
for i, intermediate_result in enumerate(processor.step_through(input_data)):
    print(f"=== After step {i} ===")

    # Inspect the observation at this stage
    obs = intermediate_result.get(TransitionKey.OBSERVATION)
    if obs:
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
```

### Interactive Debugging with Breakpoints

You can add breakpoints in the step-through loop to interactively debug:

```python
# Step through the pipeline with debugging
for i, intermediate in enumerate(processor.step_through(data)):
    print(f"Step {i}: {processor.steps[i].__class__.__name__}")

    # Set a breakpoint to inspect the current state
    breakpoint()  # Debugger will pause here

    # You can now inspect 'intermediate' in the debugger:
    # - Check tensor shapes and values
    # - Verify expected transformations
    # - Look for unexpected changes
```

During the debugger session, you can:

- Examine `intermediate[TransitionKey.OBSERVATION]` to see observation data
- Check `intermediate[TransitionKey.ACTION]` for action transformations
- Inspect any part of the transition to understand what each step does

Step-through debugging is perfect for understanding the _data_ transformations, but what about the _structure_ of that data? While hooks and step-through help you debug runtime behavior, you also need to ensure your pipeline produces data in the format expected by downstream components. This is where feature contract validation comes in.

## Validating Feature Contracts

Feature contracts define what data structure your pipeline expects as input and produces as output.
Validating these contracts helps catch mismatches early.

### Understanding Feature Contracts

Each processor step has a `transform_features()` method that describes how it changes the data structure:

```python
# Get the expected output features from your pipeline
initial_features = {
    PipelineFeatureType.OBSERVATION: {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.image": PolicyFeature(type=FeatureType.IMAGE, shape=(3, 224, 224))
    },
    PipelineFeatureType.ACTION: {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))
    }
}

# Check what your pipeline will output
output_features = processor.transform_features(initial_features)

print("Input features:")
for feature_type, features in initial_features.items():
    print(f"  {feature_type}:")
    for key, feature in features.items():
        print(f"    {key}: {feature.type.value}, shape={feature.shape}")

print("\nOutput features:")
for feature_type, features in output_features.items():
    print(f"  {feature_type}:")
    for key, feature in features.items():
        print(f"    {key}: {feature.type.value}, shape={feature.shape}")
```

### Verifying Expected Features

Check that your pipeline produces the features you expect:

```python
# Define what features you expect the pipeline to produce
expected_keys = ["observation.state", "observation.image", "action"]

print("Validating feature contract...")
for expected_key in expected_keys:
    found = False
    for feature_type, features in output_features.items():
        if expected_key in features:
            feature = features[expected_key]
            print(f"✅ {expected_key}: {feature.type.value}, shape={feature.shape}")
            found = True
            break

    if not found:
        print(f"❌ Missing expected feature: {expected_key}")
```

This validation helps ensure your pipeline will work correctly with downstream components that expect specific data structures.

## Summary

Now that you understand the three debugging approaches, you can tackle any pipeline issue systematically:

1. **Hooks** - For runtime monitoring and validation without modifying pipeline code
2. **Step-through** - For inspecting intermediate states and understanding transformations
3. **Feature validation** - For ensuring data structure contracts are met

**When to use each approach:**

- Start with **step-through debugging** when you need to understand what your pipeline does or when something unexpected happens
- Add **hooks** for continuous monitoring during development and production to catch issues automatically
- Use **feature validation** before deployment to ensure your pipeline works with downstream components

These three tools work together to give you the complete observability that complex pipelines naturally lack. With hooks watching for issues, step-through helping you understand behavior, and feature validation ensuring compatibility, you'll be able to debug any pipeline confidently and efficiently.
