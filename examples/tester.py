from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

REPO_A = "lerobot/pusht"
REPO_B = "lerobot/aloha_mobile_cabinet"  # replace with the actual repo id

feature_keys_mapping = {
    REPO_A: {  # pusht (1 camera, 2-dim)
        "action": "actions",
        "observation.state": "obs_state",
        "observation.image": "obs_image.cam_high",
    },
    REPO_B: {  # dual arm (3 cameras, 14-dim)
        "action": "actions",
        "observation.state": "obs_state",
        "observation.images.cam_high": "obs_image.cam_high",
        "observation.images.cam_left_wrist": "obs_image.cam_left_wrist",
        "observation.images.cam_right_wrist": "obs_image.cam_right_wrist",
    },
}

from torchvision.transforms.v2 import Compose, ToImage, Resize
image_tf = Compose([
    ToImage(),          # converts to tensor if needed
    Resize((224, 224)), # unify sizes across datasets (96x96 vs 480x640)
])

from torch.utils.data import DataLoader

dataset = MultiLeRobotDataset(
    repo_ids=[REPO_A, REPO_B],
    image_transforms=image_tf,              # ensures same HxW
    feature_keys_mapping=feature_keys_mapping,
    train_on_all_features=True,             # keep union of cameras; zero-fill missing
    # optional: override if you want fixed maxima; else inferred:
    # max_action_dim=14,
    # max_state_dim=14,
    max_action_dim=14,
    max_state_dim=14,
    max_image_dim=224,
    ignore_keys=[
        "next.*", # drop reward/done/success
        "index",
        "timestamp",
        "videos/*",               # drop all video metadata
        "observation.effort",     # 👈 drop effort everywhere
    ],
)
breakpoint()
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
for _ in range(100):
    batch = next(iter(loader))

breakpoint()
# vectors padded to maxima (pusht:2 -> 14; dual-arm:14 -> 14)
assert batch["actions"].shape[-1] == 14
assert batch["obs_state"].shape[-1] == 14
assert batch["actions_padding_mask"].shape[-1] == 14
assert batch["obs_state_padding_mask"].shape[-1] == 14

# cameras: all canonical keys exist; pusht will have wrists zero-filled
for cam in ["obs_image.cam_high", "obs_image.cam_left_wrist", "obs_image.cam_right_wrist"]:
    assert cam in batch
    assert f"{cam}_is_pad" in batch
    # images should all be 3x224x224 (or your transform’s size)
    img = batch[cam]
    assert img.ndim in (4, 5)  # (B,C,H,W) or (B,T,C,H,W) depending on your loader
