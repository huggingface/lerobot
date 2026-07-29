import torch

from lerobot.policies.hy_vla.processor_hy_vla import (
    dual_hy_to_native,
    dual_native_to_hy,
    dual_relative_to_current,
    dual_relative_to_first,
    dual_relative_to_native,
    matrix_to_quaternion_xyzw,
    mean_std_normalize,
    mean_std_unnormalize,
    pad_with_mask,
    quaternion_xyzw_to_matrix,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
    transform_robotwin_umi,
)


def _poses(batch: int = 2, horizon: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    pose = torch.randn(batch, horizon, 16, generator=generator, dtype=torch.float64)
    pose[..., 3:7] /= torch.linalg.vector_norm(pose[..., 3:7], dim=-1, keepdim=True)
    pose[..., 11:15] /= torch.linalg.vector_norm(pose[..., 11:15], dim=-1, keepdim=True)
    pose[..., 7] = torch.linspace(0, 1, horizon)
    pose[..., 15] = torch.linspace(1, 0, horizon)
    return pose


def test_quaternion_matrix_rotation6d_round_trip():
    pose = _poses()
    quaternion = pose[..., 3:7]
    matrix = quaternion_xyzw_to_matrix(quaternion)
    restored_matrix = quaternion_xyzw_to_matrix(matrix_to_quaternion_xyzw(matrix))
    restored_6d = rotation_6d_to_matrix(rotation_matrix_to_6d(matrix))
    torch.testing.assert_close(restored_matrix, matrix, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(restored_6d, matrix, atol=1e-10, rtol=1e-10)


def test_dual_layout_and_left_right_gripper_order_round_trip():
    pose = _poses()
    encoded = dual_native_to_hy(pose)
    assert torch.equal(encoded[..., 9], pose[..., 7])
    assert torch.equal(encoded[..., 19], pose[..., 15])
    restored = dual_hy_to_native(encoded)
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(restored[..., 3:7]),
        quaternion_xyzw_to_matrix(pose[..., 3:7]),
    )
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(restored[..., 11:15]),
        quaternion_xyzw_to_matrix(pose[..., 11:15]),
    )
    torch.testing.assert_close(
        restored[..., (0, 1, 2, 7, 8, 9, 10, 15)], pose[..., (0, 1, 2, 7, 8, 9, 10, 15)]
    )


def test_relative_first_and_current_conventions_are_explicit_and_invertible():
    pose = _poses()
    first = dual_relative_to_first(pose)
    current = dual_relative_to_current(pose, pose[:, 2])
    identity_6d = torch.tensor([1, 0, 0, 0, 1, 0], dtype=pose.dtype)
    torch.testing.assert_close(first[:, 0, 3:9], identity_6d.expand(pose.shape[0], -1))
    assert not torch.allclose(first, current)
    restored = dual_relative_to_native(first, pose[:, 0])
    torch.testing.assert_close(restored[..., :3], pose[..., :3], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(restored[..., 8:11], pose[..., 8:11], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(restored[..., 3:7]),
        quaternion_xyzw_to_matrix(pose[..., 3:7]),
        atol=1e-10,
        rtol=1e-10,
    )


def test_robotwin_umi_coordinate_and_gripper_inverse():
    pose = _poses()
    pose[..., 7] = torch.linspace(0, 1, pose.shape[-2])
    pose[..., 15] = torch.linspace(1, 0, pose.shape[-2])
    umi = transform_robotwin_umi(pose, convert_gripper=True)
    restored = transform_robotwin_umi(umi, inverse=True, convert_gripper=True)
    torch.testing.assert_close(restored[..., :3], pose[..., :3], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(restored[..., 7], pose[..., 7], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(restored[..., 15], pose[..., 15], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(restored[..., 3:7]),
        quaternion_xyzw_to_matrix(pose[..., 3:7]),
        atol=1e-10,
        rtol=1e-10,
    )


def test_padding_masks_and_zero_variance_normalization():
    value = torch.tensor([[1.0, 5.0, -2.0]])
    mean = torch.tensor([0.0, 5.0, 1.0])
    std = torch.tensor([2.0, 0.0, 4.0])
    normalized = mean_std_normalize(value, mean, std)
    assert normalized[0, 1] == 0
    restored = mean_std_unnormalize(normalized, mean, std)
    torch.testing.assert_close(restored, value)
    padded, mask = pad_with_mask(value, 5)
    torch.testing.assert_close(padded, torch.tensor([[1.0, 5.0, -2.0, 0.0, 0.0]]))
    assert torch.equal(mask, torch.tensor([[True, True, True, False, False]]))
