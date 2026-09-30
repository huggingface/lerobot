# YAM gravity model

Derived from I2RT Robotics' `yam/v1/yam.xml` and `linear_4310/linear_4310.xml`
at https://github.com/i2rt-robotics/i2rt/tree/120c3c81400171174604e503943f8d1ebc891058/i2rt/robot_models.
The MIT license is retained in `LICENSE.i2rt`.

The arm transforms, axes, limits and explicit inertial parameters are retained.
The linear gripper inertials and finger bodies replace the arm's empty gripper
placeholder. Visual meshes, contact and equality constraints are omitted: this
model is used only to compute gravity torques with zero velocity, never to
simulate collisions or certify a safe workspace. Camera and payload inertia
are not included. Only the standard YAM v1 with linear DM4310 gripper is supported.
