#!/usr/bin/env python3
"""
Unit tests for π*₀.₆ RECAP policy
"""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from lerobot.policies.pi_star_recap import (
    PiStarRECAPConfig,
    PiStarRECAPPolicy,
    ActionExpert,
    QNetwork,
    VNetwork,
    DataType,
)


@pytest.fixture
def config():
    """Default test config"""
    return PiStarRECAPConfig(
        image_size=224,
        num_obs_steps=2,
        chunk_size=4,
        max_action_dim=7,
        model=dict(
            vlm_model_name="google/paligemma-3b-pt-224",
            freeze_vlm=True,
            action_expert_hidden_size=256,
            action_expert_num_layers=2,
            action_expert_num_heads=4,
            num_q_networks=2,
            qv_hidden_size=128,
            qv_num_layers=2,
        ),
        iql=dict(
            discount=0.99,
            expectile=0.7,
            temperature=0.5,
        ),
        recap=dict(
            demo_weight=1.0,
            auto_weight=1.0,
            intervention_weight=2.0,
            use_advantage_conditioning=True,
        ),
        training=dict(
            batch_size=4,
            use_amp=False,
        ),
    )


@pytest.fixture
def mock_batch(config):
    """Create mock batch for testing"""
    batch_size = 4
    return {
        'observation.images': torch.randn(batch_size, config.num_obs_steps, 3, config.image_size, config.image_size),
        'observation.state': torch.randn(batch_size, 14),
        'action': torch.randn(batch_size, config.chunk_size, config.max_action_dim),
        'reward': torch.randn(batch_size, 1),
        'done': torch.zeros(batch_size, 1),
        'data_type': [DataType.DEMO.value] * batch_size,
    }


def test_config_creation():
    """Test config creation"""
    config = PiStarRECAPConfig()
    assert config.chunk_size > 0
    assert config.max_action_dim > 0
    assert config.iql.expectile > 0.5
    assert config.iql.discount <= 1.0


def test_config_to_dict(config):
    """Test config serialization"""
    config_dict = config.to_dict()
    assert 'model' in config_dict
    assert 'iql' in config_dict
    assert 'recap' in config_dict
    
    # Test roundtrip
    config2 = PiStarRECAPConfig.from_dict(config_dict)
    assert config2.chunk_size == config.chunk_size


def test_action_expert_forward(config):
    """Test action expert forward pass"""
    expert = ActionExpert(config)
    
    batch_size = 4
    chunk_size = config.chunk_size
    action_dim = config.max_action_dim
    hidden_size = config.model.action_expert_hidden_size
    
    noisy_actions = torch.randn(batch_size, chunk_size, action_dim)
    timestep = torch.rand(batch_size)
    context = torch.randn(batch_size, 10, hidden_size)  # VLM context
    advantage = torch.randn(batch_size, 1)
    
    velocity = expert(noisy_actions, timestep, context, advantage)
    
    assert velocity.shape == (batch_size, chunk_size, action_dim)
    assert not torch.isnan(velocity).any()


def test_q_network(config):
    """Test Q-network"""
    batch_size = 4
    hidden_size = 512  # Mock VLM hidden size
    
    q_net = QNetwork(config, hidden_size)
    
    context = torch.randn(batch_size, hidden_size)
    actions = torch.randn(batch_size, config.chunk_size, config.max_action_dim)
    
    q_value = q_net(context, actions)
    
    assert q_value.shape == (batch_size, 1)
    assert not torch.isnan(q_value).any()


def test_v_network(config):
    """Test V-network"""
    batch_size = 4
    hidden_size = 512
    
    v_net = VNetwork(config, hidden_size)
    
    context = torch.randn(batch_size, hidden_size)
    v_value = v_net(context)
    
    assert v_value.shape == (batch_size, 1)
    assert not torch.isnan(v_value).any()


def test_get_data_type_weights(config, mock_batch):
    """Test data type weight computation"""
    policy = PiStarRECAPPolicy(config)
    
    data_types = ['demo', 'auto', 'intervention', 'demo']
    weights = policy._get_data_type_weights(data_types, torch.device('cpu'))
    
    assert weights.shape == (4, 1)
    assert weights[0] == config.recap.demo_weight
    assert weights[1] == config.recap.auto_weight
    assert weights[2] == config.recap.intervention_weight


def test_compute_v_loss(config, mock_batch):
    """Test value loss computation"""
    policy = PiStarRECAPPolicy(config)
    
    context = torch.randn(4, 512)
    actions = mock_batch['action']
    type_weights = torch.ones(4, 1)
    
    v_loss = policy._compute_v_loss(context, actions, type_weights)
    
    assert v_loss.item() >= 0
    assert not torch.isnan(v_loss)


def test_compute_q_loss(config, mock_batch):
    """Test Q loss computation"""
    policy = PiStarRECAPPolicy(config)
    
    context = torch.randn(4, 512)
    actions = mock_batch['action']
    rewards = mock_batch['reward']
    dones = mock_batch['done']
    type_weights = torch.ones(4, 1)
    
    q_loss = policy._compute_q_loss(context, actions, rewards, dones, type_weights)
    
    assert q_loss.item() >= 0
    assert not torch.isnan(q_loss)


def test_compute_policy_loss(config, mock_batch):
    """Test policy loss computation"""
    policy = PiStarRECAPPolicy(config)
    
    context = torch.randn(4, 512)
    actions = mock_batch['action']
    type_weights = torch.ones(4, 1)
    
    policy_loss = policy._compute_policy_loss(context, actions, type_weights)
    
    assert policy_loss.item() >= 0
    assert not torch.isnan(policy_loss)


def test_checkpoint_save_load(config, tmp_path):
    """Test checkpoint saving and loading"""
    policy = PiStarRECAPPolicy(config)
    
    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    policy.global_step = 100
    policy.epoch = 5
    policy.save_checkpoint(str(checkpoint_path), metadata={'test': True})
    
    assert checkpoint_path.exists()
    
    # Load checkpoint
    policy2 = PiStarRECAPPolicy(config)
    metadata = policy2.load_checkpoint(str(checkpoint_path))
    
    assert policy2.global_step == 100
    assert policy2.epoch == 5
    assert metadata['test'] == True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision(config):
    """Test mixed precision training setup"""
    config.training.use_amp = True
    config.training.amp_dtype = "float16"
    
    policy = PiStarRECAPPolicy(config)
    policy = policy.cuda()
    
    assert policy.use_amp
    assert policy.scaler is not None


def test_advantage_conditioning(config):
    """Test advantage conditioning in policy"""
    config.recap.use_advantage_conditioning = True
    
    expert = ActionExpert(config)
    
    batch_size = 4
    chunk_size = config.chunk_size
    action_dim = config.max_action_dim
    hidden_size = config.model.action_expert_hidden_size
    
    noisy_actions = torch.randn(batch_size, chunk_size, action_dim)
    timestep = torch.rand(batch_size)
    context = torch.randn(batch_size, 10, hidden_size)
    advantage = torch.randn(batch_size, 1)
    
    # With advantage
    velocity_with_adv = expert(noisy_actions, timestep, context, advantage)
    
    # Without advantage
    velocity_no_adv = expert(noisy_actions, timestep, context, None)
    
    # Should be different
    assert not torch.allclose(velocity_with_adv, velocity_no_adv, atol=1e-6)


def test_target_network_sync(config):
    """Test target network synchronization"""
    policy = PiStarRECAPPolicy(config)
    
    # Modify Q-network
    for p in policy.q_networks[0].parameters():
        p.data += 1.0
    
    # Get target param before sync
    target_param_before = list(policy.q_target_networks[0].parameters())[0].data.clone()
    
    # Sync with tau=0.1
    policy._sync_target_networks(tau=0.1)
    
    # Get target param after sync
    target_param_after = list(policy.q_target_networks[0].parameters())[0].data
    
    # Should be updated
    assert not torch.allclose(target_param_before, target_param_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
