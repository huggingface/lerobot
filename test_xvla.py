from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config

cfg = make_policy_config("xvla")

dataset_id = "lerobot/svla_so101_pickplace"
# This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
dataset_metadata = LeRobotDatasetMetadata(dataset_id)
policy = make_policy(cfg=cfg, ds_meta=dataset_metadata)
print(policy)
