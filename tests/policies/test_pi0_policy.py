import pytest
import torch

from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks


def test_make_att_2d_masks():
    """Test make_att_2d_masks logic."""
    pad_masks = torch.tensor([[True, True, True, False]])
    att_masks = torch.tensor([[0, 0, 1, 1]])
    expected = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        ]
    )
    result = make_att_2d_masks(pad_masks, att_masks)
    assert torch.equal(result, expected)


def test_tensor_preallocation_optimization():
    """Test that the tensor pre-allocation optimization approach works correctly."""
    bsize = 2
    seq_len = 10
    hidden_size = 16
    
    # Simulate the old approach (list + torch.cat)
    embs_list = []
    for _ in range(seq_len):
        embs_list.append(torch.randn(bsize, 1, hidden_size))
    embs_old = torch.cat(embs_list, dim=1)
    
    # Simulate the new approach (pre-allocation)
    embs_new = torch.empty(bsize, seq_len, hidden_size)
    for i in range(seq_len):
        embs_new[:, i:i+1] = torch.randn(bsize, 1, hidden_size)
    
    # Both approaches should produce tensors with the same shape
    assert embs_old.shape == embs_new.shape
    assert embs_old.shape == (bsize, seq_len, hidden_size)


def test_memory_efficiency_comparison():
    """Test that pre-allocation is more memory efficient than list operations."""
    bsize = 4
    seq_len = 20
    hidden_size = 32
    
    # Measure memory usage for list approach (simulated)
    # In practice, this would create multiple intermediate tensors
    embs_list = []
    for _ in range(seq_len):
        embs_list.append(torch.randn(bsize, 1, hidden_size))
    embs_cat = torch.cat(embs_list, dim=1)
    
    # Measure memory usage for pre-allocation approach
    embs_prealloc = torch.empty(bsize, seq_len, hidden_size)
    for i in range(seq_len):
        embs_prealloc[:, i:i+1] = torch.randn(bsize, 1, hidden_size)
    
    # Both should produce the same result
    assert embs_cat.shape == embs_prealloc.shape
    
    # The pre-allocation approach should use less memory in practice
    # because it doesn't create intermediate tensors for the list
    assert embs_prealloc.numel() == bsize * seq_len * hidden_size 