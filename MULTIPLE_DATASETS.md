# Multiple Datasets Support in LeRobot

This document explains how to use the multiple datasets feature in LeRobot, which allows you to train policies on concatenated datasets.

## Overview

The multiple datasets feature allows you to:
- Train on multiple datasets simultaneously
- Combine datasets that share common features
- Maintain dataset indexing to track which dataset each sample comes from
- Use the same training infrastructure with minimal configuration changes

## Usage

### Single Dataset (Backwards Compatible)

The traditional way to specify a single dataset still works:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion \
    --env.type=pusht
```

### Multiple Datasets

To specify multiple datasets, pass a JSON list of repository IDs:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id='["lerobot/pusht", "lerobot/aloha_sim_insertion_human"]' \
    --policy.type=diffusion \
    --env.type=pusht
```

### Configuration File

You can also specify multiple datasets in a configuration file:

```json
{
    "dataset": {
        "repo_id": ["lerobot/pusht", "lerobot/aloha_sim_insertion_human"],
        "episodes": null,
        "image_transforms": {"enable": false},
        "revision": null,
        "use_imagenet_stats": true,
        "video_backend": "pyav"
    },
    "policy": {
        "type": "diffusion"
    }
}
```

## How It Works

### Dataset Concatenation

When multiple datasets are specified:

1. **Individual Dataset Creation**: Each dataset is created as a separate `LeRobotDataset` instance
2. **Feature Intersection**: Only features common to all datasets are kept
3. **Concatenation**: Datasets are concatenated into a `MultiLeRobotDataset`
4. **Index Mapping**: Each sample gets a `dataset_index` field indicating its source dataset

### Feature Compatibility

Only data keys that are common across all datasets are retained. If datasets have different features:

- Common features are kept
- Unique features are disabled with a warning
- Training proceeds with the intersection of features

### Episode Configuration

When using multiple datasets with episode filtering:

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id='["lerobot/dataset1", "lerobot/dataset2"]' \
    --dataset.episodes='[0, 1, 2]' \
    --policy.type=diffusion
```

The episode indices are applied to **all** datasets. Each dataset will only use episodes 0, 1, and 2.

## Implementation Details

### Dataset Index Tracking

Each sample in the concatenated dataset includes a `dataset_index` field:

```python
item = dataset[0]
print(item['dataset_index'])  # torch.tensor(0) for first dataset, torch.tensor(1) for second, etc.
```

### Repository ID Mapping

Access the mapping between repository IDs and indices:

```python
dataset = make_dataset(cfg)
print(dataset.repo_id_to_index)  # {'lerobot/pusht': 0, 'lerobot/aloha_sim_insertion_human': 1}
print(dataset.repo_index_to_id)  # {0: 'lerobot/pusht', 1: 'lerobot/aloha_sim_insertion_human'}
```

### Delta Timestamps

For policies that use temporal information:
- Delta timestamps are computed using the **first dataset's** metadata
- This assumes all datasets have similar temporal structure
- Future versions may support per-dataset delta timestamps

### Statistics Aggregation

Dataset statistics (mean, std, etc.) are aggregated across all datasets using weighted averages based on the number of samples in each dataset.

## Examples

### Example 1: Training on Multiple PushT Variants

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id='["lerobot/pusht", "lerobot/pusht_image"]' \
    --policy.type=diffusion \
    --env.type=pusht \
    --output_dir=outputs/train/multi_pusht
```

### Example 2: Training on Multiple Robot Tasks

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id='["lerobot/aloha_sim_insertion_human", "lerobot/aloha_sim_transfer_cube_human"]' \
    --policy.type=act \
    --env.type=aloha \
    --output_dir=outputs/train/multi_aloha
```

### Example 3: Using Configuration File

```bash
python lerobot/scripts/train.py \
    --config_path=configs/multi_dataset_config.json \
    --output_dir=outputs/train/multi_config
```

## Limitations and Considerations

### Feature Compatibility

- Datasets must share common features for training to work
- Datasets with completely different feature sets cannot be combined
- Image dimensions and data types must be compatible

### Memory Usage

- Multiple datasets may require more memory during initialization
- Consider the total size of all datasets when planning training

### Debugging

When debugging multi-dataset training:

1. Check feature compatibility warnings in the logs
2. Verify dataset index mapping is as expected
3. Monitor training metrics to ensure both datasets contribute

### Performance

- Initial dataset loading may take longer with multiple datasets
- Training performance should be similar to single dataset training
- Use fewer workers if memory becomes an issue

## Advanced Usage

### Weighted Sampling (Future Feature)

Currently, all datasets are concatenated with equal weighting. Future versions may support:
- Weighted sampling based on dataset size
- Custom sampling strategies
- Per-dataset episode filtering

### Custom Statistics (Future Feature)

Future versions may support:
- Per-dataset normalization statistics
- Custom aggregation strategies
- Runtime statistics updates

## Troubleshooting

### Common Issues

1. **"No common features" error**: Datasets have incompatible feature sets
2. **Memory errors**: Try reducing `num_workers` or using smaller datasets
3. **Video decoding errors**: Some datasets may have video format issues (unrelated to multi-dataset feature)

### Debugging Commands

```bash
# Check dataset compatibility
python -c "
from lerobot.common.datasets.factory import make_dataset
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.factory import make_policy_config

cfg = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id=['dataset1', 'dataset2']),
    policy=make_policy_config('diffusion')
)
dataset = make_dataset(cfg)
print('Features:', list(dataset.features.keys()))
print('Mapping:', dataset.repo_id_to_index)
"
```

## Contributing

If you encounter issues or have suggestions for the multiple datasets feature:

1. Check existing GitHub issues
2. Provide minimal reproduction examples
3. Include dataset repository IDs and error messages
4. Consider contributing improvements via pull requests

## Future Enhancements

Planned improvements include:
- Better episode management per dataset
- Weighted sampling strategies
- Per-dataset delta timestamps
- Enhanced debugging tools
- Performance optimizations
