# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/train/utils/augmentation.py. Import paths are rewritten to the LeRobot package;
# the changes run on the device of the frames, and augment_views takes the frames after their resize to the view sizes instead of the stored frames.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import math

import torch

GRAY_WEIGHTS = (0.2989, 0.587, 0.114)
CAMERA_FIELD_OF_VIEW = 60.0
CAMERA_ANGLES = ("yaw", "pitch", "roll")
CAMERA_AT_REST = {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "focal": 1.0,
    "shift_x": 0.0,
    "shift_y": 0.0,
    "radial": 0.0,
}


def draw_uniform(low: float, high: float) -> float:
    """Return one draw from the uniform distribution on [low, high] of the global generator."""
    return float(torch.empty(()).uniform_(low, high).item())


def draw_between(low: float, high: float) -> float:
    """Return one draw from [0, 1) of the global generator, scaled to [low, high] in double precision."""
    return float(low + (high - low) * torch.rand(()).item())


def draw_offset(largest: int) -> int:
    """Return one integer drawn uniformly from 0 .. ``largest`` of the global generator; a largest value of 0 takes no draw."""
    return 0 if largest == 0 else int(torch.randint(largest + 1, (1,)).item())


def draw_event(probability: float) -> bool:
    """Return whether an event of the given probability happens; a probability of 0 or 1 takes no draw."""
    return probability > 0 and (probability == 1 or float(torch.rand(())) < probability)


def draw_crop(probability: float, scale) -> tuple | None:
    """Return one random crop as (scale, horizontal offset, vertical offset), or None when it does not happen."""
    if not draw_event(probability):
        return None
    return draw_uniform(*scale), draw_uniform(0, 1), draw_uniform(0, 1)


def crop_frames(video: torch.Tensor, crop: tuple) -> torch.Tensor:
    """Return float frames [T, 3, H, W] cut to a crop of ``draw_crop`` and scaled back to their size."""
    scale, offset_x, offset_y = crop
    height, width = video.shape[-2:]
    crop_height = max(1, min(height, round(height * scale)))
    crop_width = max(1, min(width, round(width * scale)))
    if (crop_height, crop_width) == (height, width):
        return video
    top = round(offset_y * (height - crop_height))
    left = round(offset_x * (width - crop_width))
    return torch.nn.functional.interpolate(
        video[..., top : top + crop_height, left : left + crop_width],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def change_lighting(views, lighting: dict) -> list:
    """Return the float views after one random gain per colour channel, shared by the views and scaled to a mean of 1, and one soft round shadow per view."""
    gains = shadows = None
    if draw_event(lighting["gain_probability"]):
        gains = [draw_uniform(*lighting["gain"]) for _ in range(3)]
        mean = sum(gains) / 3
        gains = [gain / mean for gain in gains]
    if draw_event(lighting["shadow_probability"]):
        shadows = [
            (
                draw_uniform(0, 1),
                draw_uniform(0, 1),
                draw_uniform(*lighting["shadow_width"]),
                draw_uniform(*lighting["shadow_depth"]),
            )
            for _ in views
        ]
    changed = []
    for index, video in enumerate(views):
        if gains is not None:
            video = video * video.new_tensor(gains).view(1, 3, 1, 1)
        if shadows is not None:
            center_x, center_y, shadow_width, depth = shadows[index]
            height, width = video.shape[-2:]
            rows = torch.linspace(0, 1, height, dtype=video.dtype, device=video.device).view(height, 1)
            columns = torch.linspace(0, 1, width, dtype=video.dtype, device=video.device).view(1, width)
            distance = (columns - center_x).square() + (rows - center_y).square()
            video = video * (1 - depth * torch.exp(-distance / (2 * shadow_width**2)))
        changed.append(video.clamp(0, 1) if gains is not None or shadows is not None else video)
    return changed


def change_colors(views, color: dict) -> list:
    """Return the float views after one random brightness, contrast, saturation and gamma change shared by the views; every value of ``color`` is the largest relative change of its factor."""
    brightness, contrast, saturation, gamma = (
        draw_uniform(max(0.0, 1 - color[name]), 1 + color[name])
        for name in ("brightness", "contrast", "saturation", "gamma")
    )
    changed = []
    for video in views:
        video = video.clone()
        if brightness != 1:
            video.mul_(brightness)
        if contrast != 1:
            mean = video.mean(dim=(-3, -2, -1), keepdim=True)
            video.sub_(mean).mul_(contrast).add_(mean)
        if saturation != 1:
            gray = (video * video.new_tensor(GRAY_WEIGHTS).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
            video.sub_(gray).mul_(saturation).add_(gray)
        if gamma != 1:
            video.clamp_(0, 1).pow_(gamma)
        changed.append(video.clamp_(0, 1))
    return changed


def round_frames(video: torch.Tensor) -> torch.Tensor:
    """Return float frames in [0, 1] rounded to the nearest of the 256 values of a byte."""
    return video.mul(255).round_().clamp_(0, 255).div_(255)


def draw_view_change(settings: dict, height: int, width: int) -> dict:
    """Return the random change of one view of (height, width) pixels as float32 values: the crop (top, left, height, width), the rotation in degrees, the shift in pixels and the brightness, contrast, saturation, hue and gamma factors."""
    side = math.sqrt(draw_between(*settings["crop_area"]))
    crop_height = max(1, min(height, round(height * side)))
    crop_width = max(1, min(width, round(width * side)))
    change = {
        "crop_top": draw_offset(height - crop_height),
        "crop_left": draw_offset(width - crop_width),
        "crop_height": crop_height,
        "crop_width": crop_width,
        "rotation": draw_between(-settings["rotation"], settings["rotation"]),
        "shift_x": round(draw_between(-width * settings["translation"], width * settings["translation"])),
        "shift_y": round(draw_between(-height * settings["translation"], height * settings["translation"])),
    }
    for name in ("brightness", "contrast", "saturation"):
        change[name] = draw_between(max(0, 1 - settings[name]), 1 + settings[name])
    change["hue"] = draw_between(-settings["hue"], settings["hue"])
    change["gamma"] = draw_between(max(0, 1 - settings["gamma"]), 1 + settings["gamma"])
    return {name: torch.tensor(float(value)) for name, value in change.items()}


def draw_camera_move(camera: dict) -> dict:
    """Return one random camera move: the yaw, pitch and roll in degrees, the focal length factor, the shift of the image centre in pixels and the radial distortion."""
    move = {name: draw_uniform(-camera[name], camera[name]) for name in CAMERA_ANGLES}
    move["focal"] = draw_uniform(*camera["focal"])
    move["shift_x"] = draw_uniform(-camera["shift_x"], camera["shift_x"])
    move["shift_y"] = draw_uniform(-camera["shift_y"], camera["shift_y"])
    move["radial"] = draw_uniform(*camera["radial"])
    return move


def move_camera(video: torch.Tensor, move: dict) -> torch.Tensor:
    """Return float frames [T, 3, H, W] as seen after a camera move of ``draw_camera_move``: a pinhole camera with a 60 degree horizontal field of view turns about its centre, changes its focal length, shifts its image centre and gains a radial distortion; the frames are sampled bilinearly and mirrored at their borders."""
    frames, _, height, width = video.shape
    move = {name: torch.tensor([value]) for name, value in move.items()}
    zero, one = torch.zeros(1), torch.ones(1)
    cosine = {name: torch.cos(torch.deg2rad(move[name])) for name in CAMERA_ANGLES}
    sine = {name: torch.sin(torch.deg2rad(move[name])) for name in CAMERA_ANGLES}
    about_y = torch.stack(
        (cosine["yaw"], zero, sine["yaw"], zero, one, zero, -sine["yaw"], zero, cosine["yaw"]), -1
    )
    about_x = torch.stack(
        (one, zero, zero, zero, cosine["pitch"], -sine["pitch"], zero, sine["pitch"], cosine["pitch"]), -1
    )
    about_z = torch.stack(
        (cosine["roll"], -sine["roll"], zero, sine["roll"], cosine["roll"], zero, zero, zero, one), -1
    )
    rotation = about_z.reshape(1, 3, 3) @ about_x.reshape(1, 3, 3) @ about_y.reshape(1, 3, 3)
    focal = width / (2 * math.tan(math.radians(CAMERA_FIELD_OF_VIEW) / 2))
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    intrinsic = torch.tensor(((focal, 0, center_x), (0, focal, center_y), (0, 0, 1))).expand(1, -1, -1)
    moved = torch.zeros(1, 3, 3)
    moved[:, 0, 0] = moved[:, 1, 1] = focal * move["focal"]
    moved[:, 0, 2], moved[:, 1, 2], moved[:, 2, 2] = center_x + move["shift_x"], center_y + move["shift_y"], 1
    back = torch.linalg.inv(moved @ rotation @ torch.linalg.inv(intrinsic))
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij"
    )
    x, y = (columns - center_x) / focal, (rows - center_y) / focal
    lens = 1 + move["radial"][:, None, None] * (x.square() + y.square())
    points = torch.stack(
        (x * lens * focal + center_x, y * lens * focal + center_y, torch.ones_like(lens)), -1
    )
    source = torch.einsum("nij,nhwj->nhwi", back, points)
    depth = source[..., 2].clamp_min(1e-6)
    grid = torch.stack(
        (
            (source[..., 0] / depth + 0.5) * (2.0 / width) - 1,
            (source[..., 1] / depth + 0.5) * (2.0 / height) - 1,
        ),
        -1,
    )
    grid = grid.expand(frames, -1, -1, -1).to(video.device)
    return torch.nn.functional.grid_sample(video, grid, padding_mode="reflection", align_corners=False)


def crop_view(video: torch.Tensor, change: dict) -> torch.Tensor:
    """Return float frames [T, 3, H, W] cut to the crop of a change of ``draw_view_change`` and scaled back to their size by bilinear sampling."""
    frames, _, height, width = video.shape
    columns = change["crop_left"] + (torch.arange(width, dtype=torch.float32) + 0.5) * (
        change["crop_width"] / float(width)
    )
    rows = change["crop_top"] + (torch.arange(height, dtype=torch.float32) + 0.5) * (
        change["crop_height"] / float(height)
    )
    columns = columns.mul(2.0 / width).sub_(1)
    rows = rows.mul(2.0 / height).sub_(1)
    grid = torch.stack((columns[None].expand(height, -1), rows[:, None].expand(-1, width)), -1)
    grid = grid.expand(frames, -1, -1, -1).to(video.device)
    return torch.nn.functional.grid_sample(video, grid, padding_mode="zeros", align_corners=False)


def turn_view(video: torch.Tensor, change: dict) -> torch.Tensor:
    """Return float frames [T, 3, H, W] turned about their centre and shifted by a change of ``draw_view_change``, sampled bilinearly and black outside the frame."""
    frames, _, height, width = video.shape
    angle = torch.deg2rad(change["rotation"])
    cosine, sine = torch.cos(angle), torch.sin(angle)
    matrix = torch.zeros(2, 3)
    matrix[0, 0] = matrix[1, 1] = cosine
    matrix[0, 1], matrix[1, 0] = -sine, sine
    matrix[0, 2] = (-cosine * change["shift_x"] + sine * change["shift_y"]) * (2.0 / width)
    matrix[1, 2] = (-sine * change["shift_x"] - cosine * change["shift_y"]) * (2.0 / height)
    grid = torch.nn.functional.affine_grid(
        matrix.expand(frames, -1, -1).to(video.device), video.shape, align_corners=False
    )
    return torch.nn.functional.grid_sample(video, grid, padding_mode="zeros", align_corners=False)


def gray_frames(video: torch.Tensor) -> torch.Tensor:
    """Return the gray values [T, 1, H, W] of float RGB frames [T, 3, H, W]."""
    red, green, blue = video.unbind(dim=1)
    return (GRAY_WEIGHTS[0] * red + GRAY_WEIGHTS[1] * green + GRAY_WEIGHTS[2] * blue).unsqueeze(1)


def shift_hue(video: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Return float RGB frames [T, 3, H, W] with the hue moved by ``shift``, a share of the full colour circle."""
    red, green, blue = video.unbind(dim=1)
    value, lowest = video.max(dim=1).values, video.min(dim=1).values
    gray = value == lowest
    spread = value - lowest
    ones = torch.ones_like(value)
    saturation = spread / torch.where(gray, ones, value)
    divisor = torch.where(gray, ones, spread)
    red_part, green_part, blue_part = (
        (value - red) / divisor,
        (value - green) / divisor,
        (value - blue) / divisor,
    )
    hue = (value == red) * (blue_part - green_part)
    hue = hue + ((value == green) & (value != red)) * (2.0 + red_part - blue_part)
    hue = hue + ((value != green) & (value != red)) * (4.0 + green_part - red_part)
    hue = torch.remainder(torch.fmod(hue / 6.0 + 1.0, 1.0) + shift, 1)
    sector = torch.floor(hue * 6.0)
    rest = hue * 6.0 - sector
    sector = sector.to(torch.int64) % 6
    low = torch.clamp(value * (1.0 - saturation), 0.0, 1.0)
    falling = torch.clamp(value * (1.0 - saturation * rest), 0.0, 1.0)
    rising = torch.clamp(value * (1.0 - saturation * (1.0 - rest)), 0.0, 1.0)
    channels = (
        (value, falling, low, low, rising, value),
        (rising, value, value, falling, low, low),
        (low, low, rising, value, value, falling),
    )
    return torch.stack([torch.stack(channel).gather(0, sector[None])[0] for channel in channels], dim=1)


def recolor_view(video: torch.Tensor, change: dict) -> torch.Tensor:
    """Return float frames [T, 3, H, W] after the brightness, contrast, saturation, hue and gamma factors of a change of ``draw_view_change``; the contrast keeps the mean gray value of every frame."""
    video = (video * change["brightness"]).clamp_(0, 1)
    mean = gray_frames(video).mean(dim=(-3, -2, -1), keepdim=True)
    video = (video * change["contrast"] + mean * (1 - change["contrast"])).clamp_(0, 1)
    video = (video * change["saturation"] + gray_frames(video) * (1 - change["saturation"])).clamp_(0, 1)
    video = shift_hue(video, change["hue"])
    return torch.pow(video.clamp_(0, 1), float(change["gamma"]))


def change_views(views, view_names, settings: dict) -> list:
    """Return the float views [T, 3, H, W] of a window, first rounded to bytes, after random changes of every view on its own: a small move of the cameras named in ``camera_move``, which happens with its ``probability``, then a crop, a rotation, a shift and a colour change. With ``clean_probability`` the window keeps its frames."""
    views = [round_frames(video) for video in views]
    changes = None
    if not draw_event(settings["clean_probability"]):
        changes = [draw_view_change(settings, *video.shape[-2:]) for video in views]
    moves = {}
    camera = settings.get("camera_move")
    if camera:
        moves = {
            index: draw_camera_move(camera)
            for index, name in enumerate(view_names)
            if name in camera["views"]
        }
        if not draw_event(camera["probability"]):
            moves = dict.fromkeys(moves, CAMERA_AT_REST)
    if changes is None:
        return views
    for index, change in enumerate(changes):
        video = move_camera(views[index], moves[index]) if index in moves else views[index]
        views[index] = recolor_view(turn_view(crop_view(video, change), change), change)
    return views


def augment_views(views, names, settings: dict) -> list:
    """Apply the random image changes of a recipe to the frames of a training window at their view sizes.

    Every change is optional and drawn once for the window from the global torch generator: ``camera`` crops the
    named cameras; ``lighting`` and ``color`` change all views together and ``view_changes`` changes every view on
    its own.

    Args:
        views: The float frames [T, 3, h, w] in [0, 1] of every view.
        names: The name of every view, as ``camera`` and ``camera_move`` list them.
        settings: The ``augmentation`` of the recipe.

    Returns:
        One float tensor [T, 3, h, w] per view, rounded to the 256 values of a byte.
    """
    camera = settings.get("camera")
    cropped = [name for name in names if camera and name in camera["views"]]
    crop = draw_crop(camera["probability"], camera["scale"]) if cropped else None
    views = [
        crop_frames(video, crop) if crop and name in cropped else video
        for (name, video) in zip(names, views, strict=False)
    ]
    if "lighting" in settings:
        views = change_lighting(views, settings["lighting"])
    if "color" in settings:
        views = change_colors(views, settings["color"])
    if "view_changes" in settings:
        views = change_views(views, names, settings["view_changes"])
    return [round_frames(video) for video in views]
