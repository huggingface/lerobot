import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(x, **kwargs):
        return x


def main():
    # --- Config you can change ---
    dataset_repo = "YieumYoon/bimanual-center-basket-rblock-rlmerged-6"
    policy_repo = "YieumYoon/groot-bimanual-so100-cbasket-diffusion-003"
    batch_size = 8
    num_workers = 0  # keep 0 to avoid video loader issues at first
    # Number of sample frames to capture (for debugging sample predictions)
    sample_print = 10
    # -----------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset (local under ~/.cache/huggingface/lerobot/{repo_id})
    ds = LeRobotDataset(repo_id=dataset_repo)
    print(
        f"Loaded dataset '{dataset_repo}' "
        f"with {ds.num_frames} frames over {ds.num_episodes} episodes"
    )

    # Load trained Groot policy
    policy = GrootPolicy.from_pretrained(
        pretrained_name_or_path=policy_repo,
        strict=False,  # allow missing keys if any extra params were saved
    )
    policy.to(device)
    policy.config.device = device
    policy.eval()

    # Build Groot pre/post processors using dataset stats for min-max normalization
    dataset_stats = ds.meta.stats  # this is what comes from stats.json
    preprocessor, postprocessor = make_groot_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset_stats,
    )

    # DataLoader for batching
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    mse_sum = 0.0
    count = 0

    joint_sse = None  # sum of squared errors per joint
    total_frames = 0

    # Collect a small number of (pred, expert) samples for manual inspection
    sample_records = []  # list of tuples (pred_tensor_cpu, expert_tensor_cpu)

    # Try to get joint names from dataset metadata
    action_meta = ds.meta.features.get("action", {})
    joint_names = action_meta.get("names", None)

    print("Running offline evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Batches"), start=1):
            # Ground truth actions in env space: (B, 12)
            expert_actions = batch["action"].to(device)

            # Copy batch and remove actions before preprocessing (we want the model to predict them)
            obs_batch = dict(batch)
            obs_batch.pop("action", None)

            # Move tensor fields to the right device
            for k, v in obs_batch.items():
                if isinstance(v, torch.Tensor):
                    obs_batch[k] = v.to(device)

            # Preprocess (images + state + task → Groot inputs)
            proc_in = preprocessor(obs_batch)

            # Model prediction: (B, T, D_model)
            action_chunk = policy.predict_action_chunk(proc_in)

            # Postprocess: unnormalize + slice to env action dim → (B, 12) on CPU
            pred_actions = postprocessor(action_chunk)

            # If for any reason the postprocessor keeps a time dimension, take last step
            if pred_actions.dim() == 3:
                pred_actions = pred_actions[:, -1, :]

            # Move predictions to same device as expert actions
            pred_actions = pred_actions.to(expert_actions.device)

            # Compute per-frame MSE over joints
            diff = pred_actions - expert_actions
            squared = diff.double() ** 2  # (B, D)

            # Per-sample MSE (averaged over joints)
            mse = squared.mean(dim=-1)  # (B,)
            mse_sum += mse.sum().item()
            count += mse.shape[0]

            # Accumulate per-joint SSE
            batch_sse = squared.sum(dim=0).cpu()  # (D,)
            if joint_sse is None:
                joint_sse = torch.zeros_like(batch_sse, dtype=torch.float64)
            joint_sse += batch_sse.to(joint_sse.dtype)
            total_frames += pred_actions.shape[0]

            # Save a few sample predictions (on CPU) for manual inspection later
            if len(sample_records) < sample_print:
                # iterate samples in this batch
                for i in range(pred_actions.shape[0]):
                    if len(sample_records) >= sample_print:
                        break
                    sample_records.append(
                        (pred_actions[i].cpu().clone(), expert_actions[i].cpu().clone()))

            # Print running stats every N batches
            if batch_idx % 20 == 0:
                running_mse = mse_sum / max(count, 1)
                print(
                    f"[Batch {batch_idx}] "
                    f"batch mean MSE: {mse.mean().item():.4f}, "
                    f"running mean MSE: {running_mse:.4f}"
                )

    avg_mse = mse_sum / max(count, 1)
    avg_rmse = avg_mse ** 0.5
    print(
        f"\nOverall average per-frame action MSE on '{dataset_repo}': {avg_mse:.6f}")
    print(f"Overall average per-frame action RMSE: {avg_rmse:.6f}")

    if joint_sse is not None and total_frames > 0:
        joint_mse = joint_sse / float(total_frames)
        joint_rmse = joint_mse.sqrt()

        print("\nPer-joint RMSE:")
        for i, rmse in enumerate(joint_rmse.tolist()):
            if joint_names and i < len(joint_names):
                name = joint_names[i]
            else:
                name = f"joint_{i}"
            print(f"  {i:02d} ({name}): {rmse:.4f}")

        # Try to print normalized per-joint RMSE using dataset stats (if available)
        try:
            action_stats = dataset_stats.get(
                "action", {}) if dataset_stats is not None else {}
            action_min = action_stats.get("min", None)
            action_max = action_stats.get("max", None)
            action_std = action_stats.get("std", None)

            if action_min is not None and action_max is not None:
                min_v = torch.tensor(action_min, dtype=joint_rmse.dtype)
                max_v = torch.tensor(action_max, dtype=joint_rmse.dtype)
                rng = (max_v - min_v).clamp(min=1e-12)
                norm_rmse = joint_rmse / rng
                print("\nPer-joint RMSE normalized by (max-min):")
                for i, nrmse in enumerate(norm_rmse.tolist()):
                    if joint_names and i < len(joint_names):
                        name = joint_names[i]
                    else:
                        name = f"joint_{i}"
                    print(f"  {i:02d} ({name}): {nrmse:.6f}")
            elif action_std is not None:
                std_v = torch.tensor(
                    action_std, dtype=joint_rmse.dtype).clamp(min=1e-12)
                norm_rmse = joint_rmse / std_v
                print("\nPer-joint RMSE normalized by std:")
                for i, nrmse in enumerate(norm_rmse.tolist()):
                    if joint_names and i < len(joint_names):
                        name = joint_names[i]
                    else:
                        name = f"joint_{i}"
                    print(f"  {i:02d} ({name}): {nrmse:.6f}")
            else:
                print(
                    "\nNo action min/max/std available in dataset stats to normalize per-joint RMSE.")
        except Exception:
            print(
                "\nFailed to compute normalized RMSE from dataset stats (unexpected format).")

        # Print a few sample predictions vs expert for manual inspection
        if len(sample_records) > 0:
            print(
                f"\nSample prediction comparisons (first {len(sample_records)}):")
            for si, (pred_s, expert_s) in enumerate(sample_records):
                diff_s = (pred_s - expert_s)
                sample_mse = (diff_s.double() ** 2).mean().item()
                sample_rmse = sample_mse ** 0.5
                print(f" Sample {si}: RMSE={sample_rmse:.4f}")
                # print compact vectors
                pred_list = [float(x) for x in pred_s.numpy().tolist()]
                expert_list = [float(x) for x in expert_s.numpy().tolist()]
                diff_list = [float(x) for x in diff_s.numpy().tolist()]
                print(f"  pred:   {pred_list}")
                print(f"  expert: {expert_list}")
                print(f"  diff:   {diff_list}")


if __name__ == "__main__":
    main()
