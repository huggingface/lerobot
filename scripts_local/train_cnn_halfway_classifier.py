"""Train CNN binary classifier for halfway-task success (stage 4 done).

Positives: last 5 frames of `place_target_in_the_box` (stage idx 3) per demo —
robot has placed object in target.
Negatives:
  - pre-stage4: frames before `sparse_subtask_start_frames[3]` (scene early state)
  - post-stage4: frames after `sparse_subtask_end_frames[3]+1` (stages 5-6,
    robot moves toward cover — distractor showing "post-halfway").

Sampling 1:1 success:non, within non 50/50 pre/post. ResNet18 unfrozen, ImageNet
pretrained, image_key=observation.images.front. Eval on held-out demos via same
labeling.

Usage:
    uv run python scripts_local/train_cnn_halfway_classifier.py \\
        --train-ds local/sim_3stage_v2_full_v2_succonly_destale_tail30 \\
        --eval-ds domrachev03/sim_3stage_v2_val_fs \\
        --epochs 8 --batch 64 --pos-weight 10 --lr 2e-4 \\
        --image-key observation.images.front --ratio 1 \\
        --out outputs/cnn_halfway_v1
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
from torchvision.transforms.functional import resize as tv_resize, normalize as tv_normalize

sys.path.insert(0, "/home/dom-iva/github.com/orel/lerobot/lerobot/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset


STAGE_KEY = "place_target_in_the_box"  # stage 4 (0-indexed 3)
N_POS_FRAMES = 5  # last 5 frames of stage 4 = positives


def label_halfway_episodes(ds) -> dict[int, dict]:
    """Per ep: success = last N_POS_FRAMES of stage 4; neg = pre-stage4 + post-stage4."""
    df = ds.meta.episodes.to_pandas()
    out = {}
    for _, row in df.iterrows():
        ep = int(row['episode_index'])
        a = int(row['dataset_from_index']); b = int(row['dataset_to_index'])
        ep_len = b - a
        names_v = row.get('sparse_subtask_names')
        starts_v = row.get('sparse_subtask_start_frames')
        ends_v = row.get('sparse_subtask_end_frames')
        names = list(names_v) if names_v is not None and len(names_v) > 0 else []
        starts = [int(x) for x in (starts_v if starts_v is not None else [])]
        ends = [int(x) for x in (ends_v if ends_v is not None else [])]
        stage_idx = next((i for i, n in enumerate(names) if n == STAGE_KEY), None)
        drop = stage_idx is None or stage_idx >= len(ends)
        if not drop:
            s4_start = starts[stage_idx]
            s4_end = ends[stage_idx]
            # success: [s4_end - N_POS_FRAMES + 1, s4_end + 1)
            pos_lo = max(0, s4_end - N_POS_FRAMES + 1)
            pos_hi = s4_end + 1
            # negatives:
            #  pre: [0, s4_start)
            #  post: [s4_end + 1, ep_len)  — empty if truncated ds (no stages 5-6)
            pre_lo, pre_hi = 0, s4_start
            post_lo, post_hi = s4_end + 1, ep_len
            if pos_lo >= pos_hi or pre_lo >= pre_hi:
                drop = True
        out[ep] = {
            'ep_start': a, 'ep_end': b, 'ep_len': ep_len,
            'drop': bool(drop),
            'pos_range': (pos_lo, pos_hi) if not drop else None,
            'pre_range': (pre_lo, pre_hi) if not drop else None,
            'post_range': (post_lo, post_hi) if not drop and post_hi > post_lo else None,
        }
    return out


class HalfwayFrameDataset(Dataset):
    def __init__(self, ds, ep_meta, image_key: str = "observation.images.front",
                 ratio: int = 1, samples_per_epoch: int = 1024):
        self.ds = ds
        self.image_key = image_key
        self.ratio = ratio
        self.samples_per_epoch = samples_per_epoch
        self.pos_idxs = []
        self.neg_pre_idxs = []
        self.neg_post_idxs = []
        for ep, m in ep_meta.items():
            if m['drop']:
                continue
            for k in range(*m['pos_range']):
                self.pos_idxs.append(m['ep_start'] + k)
            for k in range(*m['pre_range']):
                self.neg_pre_idxs.append(m['ep_start'] + k)
            if m['post_range']:
                for k in range(*m['post_range']):
                    self.neg_post_idxs.append(m['ep_start'] + k)
        self.pos_idxs = np.array(self.pos_idxs)
        self.neg_pre_idxs = np.array(self.neg_pre_idxs)
        self.neg_post_idxs = np.array(self.neg_post_idxs)
        print(f"  pool: pos={len(self.pos_idxs)} neg_pre={len(self.neg_pre_idxs)} neg_post={len(self.neg_post_idxs)}")
        print(f"  sampling: 1/{ratio+1} pos, within neg 50/50 pre/post")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, i):
        is_pos = np.random.random() < (1.0 / (self.ratio + 1))
        if is_pos:
            idx = int(np.random.choice(self.pos_idxs))
            label = 1
        else:
            use_post = (np.random.random() < 0.5) and len(self.neg_post_idxs) > 0
            if use_post:
                idx = int(np.random.choice(self.neg_post_idxs))
            else:
                idx = int(np.random.choice(self.neg_pre_idxs))
            label = 0
        f = self.ds[idx]
        img = f[self.image_key].float()
        if img.shape[-1] != 224:
            img = tv_resize(img, [224, 224], antialias=True)
        img = tv_normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, label


class CNNCls(nn.Module):
    def __init__(self, n_classes: int = 2, freeze_stages: int = 0):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        for name, p in backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                p.requires_grad = True
            else:
                p.requires_grad = (freeze_stages == 0)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
        self.net = backbone

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, ds, ep_meta, device, image_key: str = "observation.images.front") -> dict:
    model.eval()
    correct = total = 0
    pos_correct = pos_total = 0
    neg_correct = neg_total = 0
    for ep, m in ep_meta.items():
        if m['drop']:
            continue
        ranges = [
            ('pos', m['pos_range'], 1),
            ('pre', m['pre_range'], 0),
        ]
        if m['post_range']:
            ranges.append(('post', m['post_range'], 0))
        for kind, (s, e), label in ranges:
            for k in range(s, e):
                f = ds[m['ep_start'] + k]
                img = f[image_key].float()
                if img.shape[-1] != 224:
                    img = tv_resize(img, [224, 224], antialias=True)
                img = tv_normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                logits = model(img.unsqueeze(0).to(device))
                pred = int(logits.argmax(dim=-1).item())
                ok = (pred == label)
                correct += ok; total += 1
                if label == 1:
                    pos_correct += ok; pos_total += 1
                else:
                    neg_correct += ok; neg_total += 1
    return {
        'overall_acc': correct / max(1, total),
        'pos_acc': pos_correct / max(1, pos_total),
        'neg_acc': neg_correct / max(1, neg_total),
        'n_total': total, 'n_pos': pos_total, 'n_neg': neg_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ds", required=True)
    ap.add_argument("--eval-ds", default=None, help="if None, hold out 20%% of train eps")
    ap.add_argument("--image-key", default="observation.images.front")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ratio", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/cnn_halfway_v1")
    ap.add_argument("--freeze-stages", type=int, default=0)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train ds...")
    train_ds_lr = LeRobotDataset(repo_id=args.train_ds)
    print(f"  train: {len(train_ds_lr)} fr, {train_ds_lr.num_episodes} eps")

    train_meta_full = label_halfway_episodes(train_ds_lr)
    if args.eval_ds is None:
        # 80/20 ep split for held-out eval
        eps_ok = [ep for ep, m in train_meta_full.items() if not m['drop']]
        eps_ok.sort()
        rng = np.random.default_rng(args.seed)
        rng.shuffle(eps_ok)
        n_eval = max(1, len(eps_ok) // 5)
        eval_eps = set(eps_ok[:n_eval])
        train_eps = set(eps_ok[n_eval:])
        train_meta = {ep: m for ep, m in train_meta_full.items() if ep in train_eps}
        eval_meta = {ep: m for ep, m in train_meta_full.items() if ep in eval_eps}
        eval_ds_lr = train_ds_lr
        print(f"  no --eval-ds; held-out split: train_eps={len(train_eps)}, eval_eps={len(eval_eps)}")
    else:
        print(f"Loading eval ds {args.eval_ds}...")
        eval_ds_lr = LeRobotDataset(repo_id=args.eval_ds)
        eval_meta = label_halfway_episodes(eval_ds_lr)
        train_meta = train_meta_full

    print(f"  train_meta kept={sum(1 for m in train_meta.values() if not m['drop'])} dropped={sum(1 for m in train_meta.values() if m['drop'])}")
    print(f"  eval_meta kept={sum(1 for m in eval_meta.values() if not m['drop'])} dropped={sum(1 for m in eval_meta.values() if m['drop'])}")

    train_set = HalfwayFrameDataset(train_ds_lr, train_meta, args.image_key, ratio=args.ratio, samples_per_epoch=args.samples_per_epoch)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nBuilding model (freeze_stages={args.freeze_stages})...")
    model = CNNCls(n_classes=2, freeze_stages=args.freeze_stages).to(args.device)
    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    print("\nInitial eval...")
    m = evaluate(model, eval_ds_lr, eval_meta, args.device, args.image_key)
    print(f"  init: overall={m['overall_acc']:.3f} pos={m['pos_acc']:.3f} neg={m['neg_acc']:.3f} (n_pos={m['n_pos']} n_neg={m['n_neg']})")

    print("\nTraining...")
    best_overall = 0.0
    best_pos = 0.0
    class_w = torch.tensor([1.0, float(args.pos_weight)], device=args.device)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        losses, accs = [], []
        for img, label in train_loader:
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            logits = model(img)
            loss = F.cross_entropy(logits, label, weight=class_w)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(dim=-1) == label).float().mean().item())
        dt = time.time() - t0
        m = evaluate(model, eval_ds_lr, eval_meta, args.device, args.image_key)
        print(f"  epoch {ep+1}/{args.epochs} ({dt:.1f}s): train_loss={float(np.mean(losses)):.3f} train_acc={float(np.mean(accs)):.3f}  | eval overall={m['overall_acc']:.3f} pos={m['pos_acc']:.3f} neg={m['neg_acc']:.3f}")
        if m['overall_acc'] > best_overall:
            best_overall = m['overall_acc']
            best_pos = m['pos_acc']
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"    >>> saved best.pt (overall={best_overall:.3f})")

    print(f"\nBest overall_acc={best_overall:.3f} pos_acc={best_pos:.3f}; ckpt: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
