from __future__ import annotations

from typing import Tuple
import numpy as np
import torch

from .contracts import ActionContract


def _is_torch(x):
    return isinstance(x, torch.Tensor)


def _as_array(x):
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_tensor(x, like):
    arr = _as_array(x)
    t = torch.as_tensor(arr, dtype=like.dtype if hasattr(like, "dtype") else torch.float32)
    if hasattr(like, "device"):
        t = t.to(like.device)
    return t


def to_canonical_action(action, source: ActionContract, semantics: str = "per_step"):
    """Convert *action* described by *source* into canonical physical per-step delta.

    Supports numpy arrays and torch tensors. Action shape must have last dim == 7.
    """
    is_torch = _is_torch(action)
    a = _as_array(action)
    if a.shape == ():
        raise ValueError("Invalid action shape")
    if a.shape[-1] != source.action_dim:
        raise ValueError("Last dim must equal action_dim")

    # extract slices
    trans = a[..., source.translation_dim[0] : source.translation_dim[1]]
    rot = a[..., source.rotation_dim[0] : source.rotation_dim[1]]
    grip = a[..., source.gripper_index]

    # handle representations
    if source.representation == "controller_command":
        if source.controller_translation_scale is None or source.controller_rotation_scale is None:
            raise ValueError("controller scales required for controller_command representation")
        # apply controller scales (normalized -> physical per-step)
        trans_scale = np.asarray(source.controller_translation_scale)
        rot_scale = np.asarray(source.controller_rotation_scale)
        canonical_trans = trans * trans_scale
        canonical_rot = rot * rot_scale
        canonical = np.concatenate([canonical_trans, canonical_rot, grip[..., None]], axis=-1)
        if semantics == "velocity":
            if source.fps is None:
                raise ValueError("fps required for velocity semantics")
            canonical[..., :6] = canonical[..., :6] / source.fps
    elif source.representation == "physical_delta":
        canonical = a
        if semantics == "velocity":
            if source.fps is None:
                raise ValueError("fps required for velocity semantics")
            canonical = canonical.copy()
            canonical[..., :6] = canonical[..., :6] * source.fps
    elif source.representation == "physical_velocity":
        if source.fps is None:
            raise ValueError("fps required for velocity semantics")
        # convert velocity -> per-step delta
        canonical = a.copy()
        canonical[..., :6] = canonical[..., :6] / source.fps
    else:
        raise ValueError(f"Unknown source representation: {source.representation}")

    return _as_tensor(canonical, action) if is_torch else canonical


def from_canonical_action(canonical, target: ActionContract, semantics: str = "per_step", clip: bool = False):
    is_torch = _is_torch(canonical)
    c = _as_array(canonical)
    if c.shape[-1] != target.action_dim:
        raise ValueError("Last dim must equal action_dim")

    trans = c[..., target.translation_dim[0] : target.translation_dim[1]]
    rot = c[..., target.rotation_dim[0] : target.rotation_dim[1]]
    grip = c[..., target.gripper_index]

    if target.representation == "controller_command":
        if target.controller_translation_scale is None or target.controller_rotation_scale is None:
            raise ValueError("controller scales required for controller_command representation")
        trans_scale = np.asarray(target.controller_translation_scale)
        rot_scale = np.asarray(target.controller_rotation_scale)
        out_trans = trans / trans_scale
        out_rot = rot / rot_scale
        out = np.concatenate([out_trans, out_rot, grip[..., None]], axis=-1)
        if semantics == "velocity":
            if target.fps is None:
                raise ValueError("fps required for velocity semantics")
            out[..., :6] = out[..., :6] * target.fps
    elif target.representation == "physical_delta":
        out = c
        if semantics == "velocity":
            if target.fps is None:
                raise ValueError("fps required for velocity semantics")
            out = out.copy()
            out[..., :6] = out[..., :6] / target.fps
    elif target.representation == "physical_velocity":
        if target.fps is None:
            raise ValueError("fps required for velocity semantics")
        out = c.copy()
        out[..., :6] = out[..., :6] * target.fps
    else:
        raise ValueError(f"Unknown target representation: {target.representation}")

    if clip and target.representation == "controller_command":
        out = np.clip(out, -1.0, 1.0)

    return _as_tensor(out, canonical) if is_torch else out


def convert_action(action, source: ActionContract, target: ActionContract, *, semantics: str = "per_step", clip: bool = False):
    """Convert *action* from *source* contract to *target* contract.

    Implements both 'per_step' and 'velocity' semantics matching historical behavior.
    """
    is_torch = _is_torch(action)
    a = _as_array(action)
    if a.shape[-1] != source.action_dim:
        raise ValueError("Last dim must equal action_dim")

    # split
    trans = a[..., source.translation_dim[0] : source.translation_dim[1]]
    rot = a[..., source.rotation_dim[0] : source.rotation_dim[1]]
    grip = a[..., source.gripper_index]

    # helpers to get controller scales
    def _to_array(x):
        return _as_array(x)

    # compute either per-step delta or physical velocity from source
    if semantics == "per_step":
        if source.representation == "controller_command":
            trans_scale = _to_array(source.controller_translation_scale)
            rot_scale = _to_array(source.controller_rotation_scale)
            phys_trans = trans * trans_scale
            phys_rot = rot * rot_scale
        elif source.representation == "physical_delta":
            phys_trans = trans
            phys_rot = rot
        elif source.representation == "physical_velocity":
            if source.fps is None:
                raise ValueError("fps required for velocity semantics on source")
            phys_trans = trans / source.fps
            phys_rot = rot / source.fps
        else:
            raise ValueError(f"Unknown source representation: {source.representation}")

        # now encode to target
        if target.representation == "controller_command":
            t_scale = _to_array(target.controller_translation_scale)
            r_scale = _to_array(target.controller_rotation_scale)
            out_trans = phys_trans / t_scale
            out_rot = phys_rot / r_scale
            out = np.concatenate([out_trans, out_rot, grip[..., None]], axis=-1)
            if clip:
                out = np.clip(out, -1.0, 1.0)
        elif target.representation == "physical_delta":
            out = np.concatenate([phys_trans, phys_rot, grip[..., None]], axis=-1)
        elif target.representation == "physical_velocity":
            if target.fps is None:
                raise ValueError("fps required for velocity semantics on target")
            vel_trans = phys_trans * target.fps
            vel_rot = phys_rot * target.fps
            out = np.concatenate([vel_trans, vel_rot, grip[..., None]], axis=-1)
        else:
            raise ValueError(f"Unknown target representation: {target.representation}")

    elif semantics == "velocity":
        # sanity: velocity semantics require fps information somewhere to be meaningful
        if getattr(source, "fps", None) is None and getattr(target, "fps", None) is None:
            raise ValueError("fps is required on source or target to perform 'velocity' semantics")
        # compute physical velocity from source
        if source.representation == "controller_command":
            trans_scale = _to_array(source.controller_translation_scale)
            rot_scale = _to_array(source.controller_rotation_scale)
            phys_vel_trans = trans * trans_scale * (source.fps or 1.0)
            phys_vel_rot = rot * rot_scale * (source.fps or 1.0)
        elif source.representation == "physical_delta":
            if source.fps is None:
                raise ValueError("fps required to interpret physical_delta as velocity")
            phys_vel_trans = trans * (source.fps)
            phys_vel_rot = rot * (source.fps)
        elif source.representation == "physical_velocity":
            phys_vel_trans = trans
            phys_vel_rot = rot
        else:
            raise ValueError(f"Unknown source representation: {source.representation}")

        # encode from physical velocity to target
        if target.representation == "controller_command":
            # target command = (phys_vel / target.fps) / target_scale
            t_scale = _to_array(target.controller_translation_scale)
            r_scale = _to_array(target.controller_rotation_scale)
            delta_trans = phys_vel_trans / (target.fps or 1.0)
            delta_rot = phys_vel_rot / (target.fps or 1.0)
            out_trans = delta_trans / t_scale
            out_rot = delta_rot / r_scale
            out = np.concatenate([out_trans, out_rot, grip[..., None]], axis=-1)
            if clip:
                out = np.clip(out, -1.0, 1.0)
        elif target.representation == "physical_delta":
            delta_trans = phys_vel_trans / (target.fps or 1.0)
            delta_rot = phys_vel_rot / (target.fps or 1.0)
            out = np.concatenate([delta_trans, delta_rot, grip[..., None]], axis=-1)
        elif target.representation == "physical_velocity":
            out = np.concatenate([phys_vel_trans, phys_vel_rot, grip[..., None]], axis=-1)
        else:
            raise ValueError(f"Unknown target representation: {target.representation}")

    else:
        raise ValueError(f"Unknown conversion semantics: {semantics!r}")

    return _as_tensor(out, action) if is_torch else out
