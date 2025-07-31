"""Default configuration values for dataset conversion."""

DEFAULT_CONFIG = {
    "repo_id": "arclabmit/spacecraft_moonlanding_dataset",
    "fps": 1,
    "robot_type": "spacecraft",
    "task_name": "Land on the Moon",
    "input_dir": "/home/demo/Alex/MetaRL Paper/landing_trajectories/trajectories",
    "output_dir": "./converted_datasets/spacecraft_moonlanding_dataset",
    "csv_pattern": "states/trajectory_{episode}.csv",
    "image_pattern": "imgs/img_traj_{episode}_step_{frame}",
    "image_extension": ".png",
    "action_columns": ["Tx", "Ty", "Tz", "Lx", "Ly", "Lz"],
    "state_columns": ["x", "y", "z", "vx", "vy", "vz", "mass", "q0", "q1", "q2", "q3", "w1", "w2", "w3"],
    "image_keys": ["observation.images.camera"],
    "use_videos": False,
    "debug": True,
    "test_mode": False,
    "push_to_hub": True,
    "private_repo": False,
    "tolerance_s": 10.0
}
