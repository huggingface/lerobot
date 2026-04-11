# Flow Matching Policy

This directory contains the implementation of the Flow Matching Policy, designed to follow the standard lerobot interface.

## Files
- configuration_flow_matching.py: Contains the FlowMatchingConfig dataclass storing hyperparameters.
- modeling_flow_matching.py: Contains the core mathematical implementation (ODE solver, vector field network) and FlowMatchingPolicy module interface.
- processor_flow_matching.py: Contains pre/post-processing pipelines for Flow Matching models. Data transforms and normalization.
