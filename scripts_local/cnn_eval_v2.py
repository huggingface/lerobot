"""Quick CNN P(success) sanity check on v2 dataset.

Loads outputs/cnn_success_cls/best.pt and scores last + first frames of
successful eps from domrachev03/sim_3stage_v2_train_fs.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
import torch.nn as nn
import imageio.v2 as imageio
import pyarrow.parquet as pq
import pandas as pd
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from torchvision import transforms

DEVICE = "cuda"


class CNNCls(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
        self.net = backbone

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="domrachev03/sim_3stage_v2_train_fs")
    ap.add_argument("--ckpt", default="outputs/cnn_success_cls/best.pt")
    ap.add_argument("--n-eps", type=int, default=15)
    ap.add_argument("--image-key", default="observation.images.wrist")
    args = ap.parse_args()

    model = CNNCls().to(DEVICE)
    sd = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    root = Path(HF_LEROBOT_HOME) / args.repo_id
    eps = pd.concat([pq.read_table(p).to_pandas() for p in sorted((root/"meta/episodes/chunk-000").glob("file-*.parquet"))], ignore_index=True)
    df = pd.concat([pq.read_table(p).to_pandas() for p in sorted((root/"data/chunk-000").glob("file-*.parquet"))], ignore_index=True)

    succ_eps = []
    for _, r in eps.iterrows():
        s = df.iloc[int(r["dataset_from_index"]):int(r["dataset_to_index"])]
        if (s["next.reward"] >= 0.5).any():
            succ_eps.append(int(r["episode_index"]))
    succ_eps = succ_eps[:args.n_eps]
    print(f"using {len(succ_eps)} successful eps")

    @torch.no_grad()
    def score(img_np):
        x = tx(img_np).unsqueeze(0).to(DEVICE)
        logits = model(x)
        return float(F.softmax(logits, dim=-1)[0, 1].cpu())

    rows = []
    for ep_idx in succ_eps:
        r = eps[eps["episode_index"] == ep_idx].iloc[0]
        cam = args.image_key
        chunk_idx = int(r[f"videos/{cam}/chunk_index"])
        file_idx = int(r[f"videos/{cam}/file_index"])
        from_t = float(r[f"videos/{cam}/from_timestamp"])
        T = int(r["length"])
        vid = root / "videos" / cam / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        rd = imageio.get_reader(str(vid))
        fps = 20
        f0 = int(round(from_t * fps))
        # first 5 frames
        rd.set_image_index(f0)
        first_probs = [score(rd.get_next_data()) for _ in range(5)]
        # last 5 frames
        rd.set_image_index(f0 + T - 5)
        last_probs = [score(rd.get_next_data()) for _ in range(5)]
        rows.append({
            "ep": ep_idx,
            "T": T,
            "first5_max": max(first_probs),
            "first5_mean": np.mean(first_probs),
            "last5_max": max(last_probs),
            "last5_mean": np.mean(last_probs),
        })
        print(f"ep{ep_idx:3d} T={T:3d}  first5(mean/max)={np.mean(first_probs):.3f}/{max(first_probs):.3f}  last5(mean/max)={np.mean(last_probs):.3f}/{max(last_probs):.3f}")

    print()
    print(f"all-eps last5_max>0.5 rate: {sum(1 for r in rows if r['last5_max']>0.5)/len(rows)*100:.0f}% ({sum(1 for r in rows if r['last5_max']>0.5)}/{len(rows)})")
    print(f"all-eps last5_max>0.9 rate: {sum(1 for r in rows if r['last5_max']>0.9)/len(rows)*100:.0f}% ({sum(1 for r in rows if r['last5_max']>0.9)}/{len(rows)})")
    print(f"all-eps first5_max>0.5 rate (FP): {sum(1 for r in rows if r['first5_max']>0.5)/len(rows)*100:.0f}%")


if __name__ == "__main__":
    main()
