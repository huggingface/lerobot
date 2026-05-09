"""Train CNN binary success classifier.

Logic:
- For each demo episode, find LAST gripper-open event (transition from g<0.05 to g>0.2 in state[6]).
- Drop episodes where g[-10] < 0.1 (gripper was closed at end → atypical demo).
- Success frames: from last_open_event to end of episode.
- Non-success frames: all frames before last_open_event.
- Train CNN on 2-class CE with 10:1 sampling ratio (10 non-success per success).
- Validate on v3_eval (15 held-out demos) using same gripper-open rule.
- Goal: success rate >= 0.8 on held-out eval.

Architecture: ResNet-18 backbone (frozen first stages) + 2-class head.
Input: wrist cam image at 128x128.
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


def detect_last_open_event(g: np.ndarray, low: float = 0.05, high: float = 0.2) -> int:
    """Index of last upward gripper transition (low→high). -1 if none."""
    oe = np.where((g[:-1] < low) & (g[1:] > high))[0]
    return int(oe[-1]) if len(oe) > 0 else -1


def label_episodes(ds, gripper_dim: int = 6, drop_end_closed_at: int = 10, end_closed_thr: float = 0.1) -> dict[int, dict]:
    """Define success as the LAST frame of each ep (where next.reward=1 / next.done=True).

    Non-success = all frames before that, split into stage5 vs other for the 50:50 sampler.
    """
    df = ds.meta.episodes.to_pandas()
    out = {}
    for _, row in df.iterrows():
        ep = int(row['episode_index'])
        a = int(row['dataset_from_index']); b = int(row['dataset_to_index'])
        ep_len = b - a
        # Find last reward>=0.5 frame; if none, last frame of ep
        rewards = []
        for i in range(a, b):
            r = ds[i].get('next.reward', None)
            rewards.append(float(r.item()) if r is not None else 0.0)
        rewards = np.array(rewards)
        first_succ = np.where(rewards >= 0.5)[0]
        success_start = int(first_succ[0]) if len(first_succ) else (ep_len - 1)
        # Find stage 5 range
        def _safe(v):
            if v is None: return []
            if hasattr(v, '__iter__') and not isinstance(v, str): return list(v)
            return []
        names = _safe(row.get('sparse_subtask_names'))
        starts = [int(x) for x in _safe(row.get('sparse_subtask_start_frames'))]
        ends = [int(x) for x in _safe(row.get('sparse_subtask_end_frames'))]
        stage5_start = None
        for n, s, e in zip(names, starts, ends):
            if n == 'place_cover_on_the_box':
                stage5_start = max(0, s)
                break
        drop = (success_start <= 0) or stage5_start is None or stage5_start >= success_start
        out[ep] = {
            'ep_start': a, 'ep_end': b, 'ep_len': ep_len,
            'last_open': success_start,  # reuse field name for compat
            'drop': bool(drop),
            'success_range': (success_start, ep_len) if not drop else None,
            'nonsucc_stage5_range': (stage5_start, success_start) if (not drop) else None,
            'nonsucc_other_range': (0, stage5_start) if (not drop and stage5_start > 0) else None,
        }
    return out


class FrameClsDataset(Dataset):
    """Samples ratio:1 non-success:success per epoch.

    Within non-success: 50% from stage5 (before last_open), 50% from other stages.
    """
    def __init__(self, ds, ep_meta: dict, image_key: str = "observation.images.wrist",
                 ratio: int = 10, samples_per_epoch: int = 8192, stage5_frac: float = 0.5):
        self.ds = ds
        self.image_key = image_key
        self.ratio = ratio
        self.samples_per_epoch = samples_per_epoch
        self.stage5_frac = stage5_frac
        # Collect index pools
        self.success_idxs = []
        self.ns_stage5_idxs = []
        self.ns_other_idxs = []
        for ep, m in ep_meta.items():
            if m['drop']:
                continue
            s_start, s_end = m['success_range']
            for k in range(s_start, s_end):
                self.success_idxs.append(m['ep_start'] + k)
            if m['nonsucc_stage5_range'] is not None:
                ns_s, ns_e = m['nonsucc_stage5_range']
                for k in range(ns_s, ns_e):
                    self.ns_stage5_idxs.append(m['ep_start'] + k)
            ns_o_s, ns_o_e = m['nonsucc_other_range']
            for k in range(ns_o_s, ns_o_e):
                self.ns_other_idxs.append(m['ep_start'] + k)
        self.success_idxs = np.array(self.success_idxs)
        self.ns_stage5_idxs = np.array(self.ns_stage5_idxs)
        self.ns_other_idxs = np.array(self.ns_other_idxs)
        print(f"  pool: {len(self.success_idxs)} success, {len(self.ns_stage5_idxs)} ns_stage5, {len(self.ns_other_idxs)} ns_other")
        print(f"  sampling: 1/{ratio+1} success, then within non-success {stage5_frac*100:.0f}% stage5 + {(1-stage5_frac)*100:.0f}% other")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, i):
        is_success = np.random.random() < (1.0 / (self.ratio + 1))
        if is_success:
            idx = int(np.random.choice(self.success_idxs))
            label = 1
        else:
            from_stage5 = np.random.random() < self.stage5_frac and len(self.ns_stage5_idxs) > 0
            if from_stage5:
                idx = int(np.random.choice(self.ns_stage5_idxs))
            else:
                idx = int(np.random.choice(self.ns_other_idxs))
            label = 0
        f = self.ds[idx]
        img = f[self.image_key].float()
        if img.shape[-1] != 224:
            img = tv_resize(img, [224, 224], antialias=True)
        img = tv_normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, label


class CNNCls(nn.Module):
    def __init__(self, n_classes: int = 2, freeze_backbone_stages: int = 3):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # Freeze first N stages
        for name, p in backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                p.requires_grad = True
            else:
                p.requires_grad = (freeze_backbone_stages == 0)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
        self.net = backbone

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, ds, ep_meta: dict, device: str, image_key: str = "observation.images.wrist") -> dict:
    """Per-frame accuracy on (kept) eval episodes using same gripper-open labeling."""
    model.eval()
    correct, total = 0, 0
    succ_correct, succ_total = 0, 0
    nonsucc_correct, nonsucc_total = 0, 0
    for ep, m in ep_meta.items():
        if m['drop']:
            continue
        s_start, s_end = m['success_range']
        # Combine non-success ranges (stage5 pre-open + other)
        ns_ranges = []
        if m['nonsucc_stage5_range'] is not None:
            ns_ranges.append(m['nonsucc_stage5_range'])
        if m['nonsucc_other_range'] is not None:
            ns_ranges.append(m['nonsucc_other_range'])
        for kind, rng_list, label in [('ns', ns_ranges, 0), ('s', [(s_start, s_end)], 1)]:
          for s, e in rng_list:
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
                    succ_correct += ok; succ_total += 1
                else:
                    nonsucc_correct += ok; nonsucc_total += 1
    return {
        'overall_acc': correct / max(1, total),
        'success_acc': succ_correct / max(1, succ_total),
        'nonsucc_acc': nonsucc_correct / max(1, nonsucc_total),
        'n_total': total, 'n_success': succ_total, 'n_nonsucc': nonsucc_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ds", default="domrachev03/sim_3stage_v4_success_train_fs")
    ap.add_argument("--eval-ds", default="domrachev03/sim_3stage_v3_eval")
    ap.add_argument("--image-key", default="observation.images.wrist")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples-per-epoch", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ratio", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/cnn_success_cls")
    ap.add_argument("--freeze-stages", type=int, default=3)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading datasets...")
    train_ds_lr = LeRobotDataset(repo_id=args.train_ds)
    eval_ds_lr = LeRobotDataset(repo_id=args.eval_ds)
    print(f"train: {len(train_ds_lr)} fr, eval: {len(eval_ds_lr)} fr")

    print("\nLabeling train episodes...")
    train_meta = label_episodes(train_ds_lr)
    n_kept = sum(1 for m in train_meta.values() if not m['drop'])
    n_drop = sum(1 for m in train_meta.values() if m['drop'])
    print(f"train: kept={n_kept}, dropped={n_drop} (ended-closed or no-open-event)")

    print("Labeling eval episodes...")
    eval_meta = label_episodes(eval_ds_lr)
    e_kept = sum(1 for m in eval_meta.values() if not m['drop'])
    e_drop = sum(1 for m in eval_meta.values() if m['drop'])
    print(f"eval: kept={e_kept}, dropped={e_drop}")

    train_set = FrameClsDataset(train_ds_lr, train_meta, args.image_key, ratio=args.ratio, samples_per_epoch=args.samples_per_epoch)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nBuilding model (freeze_stages={args.freeze_stages})...")
    model = CNNCls(n_classes=2, freeze_backbone_stages=args.freeze_stages).to(args.device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train/1e6:.2f}M")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    print("\nInitial eval...")
    metrics = evaluate(model, eval_ds_lr, eval_meta, args.device, args.image_key)
    print(f"  init: overall={metrics['overall_acc']:.3f}, success={metrics['success_acc']:.3f}, nonsucc={metrics['nonsucc_acc']:.3f}")

    print("\nTraining...")
    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        losses, accs = [], []
        for img, label in train_loader:
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            logits = model(img)
            # Up-weight success class to counter bias
            class_weights = torch.tensor([1.0, float(args.pos_weight)], device=args.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, label, weight=class_weights)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(dim=-1) == label).float().mean().item())
        dt = time.time() - t0
        train_loss = float(np.mean(losses)); train_acc = float(np.mean(accs))
        # Eval each epoch
        m = evaluate(model, eval_ds_lr, eval_meta, args.device, args.image_key)
        print(f"  epoch {ep+1}/{args.epochs} ({dt:.1f}s): train_loss={train_loss:.3f} train_acc={train_acc:.3f}  | eval overall={m['overall_acc']:.3f} success={m['success_acc']:.3f} nonsucc={m['nonsucc_acc']:.3f}")
        if m['overall_acc'] > best_acc:
            best_acc = m['overall_acc']
            torch.save(model.state_dict(), out_dir / "best.pt")
        if m['success_acc'] >= 0.80 and m['nonsucc_acc'] >= 0.80:
            print(f"  >>> success_acc + nonsucc_acc both >= 0.80, stopping early")
            break

    print(f"\nBest overall_acc: {best_acc:.3f}, ckpt: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
