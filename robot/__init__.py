"""
Robot manipulation system — high-level wrappers for LeRobot + OAK-D.

Modules:
    arm_controller: ArmController abstraction (move_to_pose, set_gripper, home, e-stop)
    dataset_collector: Episode collection for policy training
    train: Training wrapper for ACT / Diffusion Policy
    evaluate: Policy evaluation on real robot
"""
