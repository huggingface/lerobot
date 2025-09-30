#!/usr/bin/env python
"""
Debug script to check normalization configuration and stats.
Run this with your training configuration to see what's happening with normalization.
"""

import torch
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@parser.wrap()
def debug_normalization(cfg: TrainPipelineConfig):
    print("\n" + "="*80)
    print("NORMALIZATION DEBUG REPORT")
    print("="*80)
    
    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = make_dataset(cfg)
    print(f"   Dataset: {cfg.dataset.repo_id}")
    print(f"   Frames: {dataset.num_frames}")
    print(f"   Episodes: {dataset.num_episodes}")
    
    # Check dataset stats
    print("\n📊 Dataset Stats Keys:")
    if dataset.meta.stats:
        for key in sorted(dataset.meta.stats.keys()):
            stats = dataset.meta.stats[key]
            stat_keys = list(stats.keys())
            print(f"   '{key}': {stat_keys}")
            
            # Check if quantile stats are present
            has_quantiles = 'q01' in stats and 'q99' in stats
            has_quantile10 = 'q10' in stats and 'q90' in stats
            has_min_max = 'min' in stats and 'max' in stats
            has_mean_std = 'mean' in stats and 'std' in stats
            
            if has_quantiles:
                print(f"      ✅ Has q01/q99 quantiles: [{stats['q01']}, {stats['q99']}]")
            else:
                print(f"      ❌ Missing q01/q99 quantiles")
            
            if has_quantile10:
                print(f"      ✅ Has q10/q90 quantiles")
            
            if has_min_max:
                print(f"      ✅ Has min/max: [{stats['min']}, {stats['max']}]")
            
            if has_mean_std:
                print(f"      ✅ Has mean/std")
    else:
        print("   ⚠️  No stats found in dataset!")
    
    # Load policy
    print("\n🤖 Loading policy...")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    print(f"   Policy type: {cfg.policy.type}")
    print(f"   Pretrained path: {cfg.policy.pretrained_path}")
    
    # Check policy features
    print("\n🎯 Policy Input Features:")
    for key, feature in policy.config.input_features.items():
        print(f"   '{key}': type={feature.type.value}, shape={feature.shape}")
    
    print("\n🎯 Policy Output Features:")
    for key, feature in policy.config.output_features.items():
        print(f"   '{key}': type={feature.type.value}, shape={feature.shape}")
    
    # Check normalization mapping
    print("\n🔧 Policy Normalization Mapping:")
    if hasattr(policy.config, 'normalization_mapping') and policy.config.normalization_mapping:
        for feature_type, norm_mode in policy.config.normalization_mapping.items():
            print(f"   {feature_type.value}: {norm_mode.value}")
    else:
        print("   ⚠️  No normalization mapping found!")
    
    # Create processors
    print("\n⚙️  Creating processors...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats, 
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )
    
    # Check preprocessor steps
    print("\n🔄 Preprocessor Steps:")
    for i, step in enumerate(preprocessor.steps):
        print(f"   {i+1}. {step.__class__.__name__}")
        if hasattr(step, 'features'):
            print(f"      Features: {list(step.features.keys())}")
        if hasattr(step, 'norm_map'):
            print(f"      Norm map: {step.norm_map}")
        if hasattr(step, 'stats') and step.stats:
            print(f"      Stats keys: {list(step.stats.keys())}")
        if hasattr(step, '_tensor_stats') and step._tensor_stats:
            print(f"      Tensor stats keys: {list(step._tensor_stats.keys())}")
    
    # Test with a sample batch
    print("\n🧪 Testing with sample batch...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    
    print("\n   Input batch keys:")
    for key in sorted(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            print(f"      '{key}': shape={batch[key].shape}, dtype={batch[key].dtype}")
    
    # Process batch
    print("\n   Processing batch...")
    processed_batch = preprocessor(batch)
    
    print("\n   Processed batch keys:")
    for key in sorted(processed_batch.keys()):
        if isinstance(processed_batch[key], torch.Tensor):
            tensor = processed_batch[key]
            print(f"      '{key}': shape={tensor.shape}, min={tensor.min().item():.3f}, max={tensor.max().item():.3f}, mean={tensor.mean().item():.3f}")
            
            # Check if normalized properly
            if tensor.min().item() < -1.1 or tensor.max().item() > 1.1:
                print(f"         ⚠️  WARNING: Values outside [-1, 1] range!")
    
    print("\n" + "="*80)
    print("END OF DEBUG REPORT")
    print("="*80 + "\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    debug_normalization()
