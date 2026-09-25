# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Vendored from black-forest-labs/flux-action (src/flux_action/models/positional.py).
"""Position-id packers and scatter helpers."""

from collections.abc import Callable

import torch
from einops import rearrange
from torch import Tensor


def prc_img(x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, h, w = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(h, device=dev),
        "w": torch.arange(w, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids


def prc_vid(x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, t, h, w = x.shape
    dev = x.device
    if t_coord is None:
        t_coord = torch.arange(t, device=dev)
    coords = {
        "t": t_coord.to(dev),
        "h": torch.arange(h, device=dev),
        "w": torch.arange(w, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "c t h w -> (t h w) c")
    return x, x_ids


def prc_audio(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    _, t = x.shape
    dev = x.device
    t_coord = torch.arange(t, device=dev) if t_coord is None else t_coord.to(dev)
    coords = {
        "t": t_coord,
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "c t -> t c")
    return x, x_ids


def prc_txt(x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txt"
    length, _ = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(length, device=dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids


def prc_txts(
    x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txts"
    t, length, _ = x.shape
    dev = x.device
    if t_coord is None:
        t_coord = torch.arange(t, device=dev)
    coords = {
        "t": t_coord.to(dev),
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(length, device=dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "t l c -> (t l) c")
    return x, x_ids


def _batched(fn: Callable) -> Callable:
    def wrapped(
        x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        pairs = [
            fn(
                x[i],
                t_coord[i] if t_coord is not None else None,
                l_coord[i] if l_coord is not None else None,
            )
            for i in range(len(x))
        ]
        xs, ids = zip(*pairs, strict=True)
        return torch.stack(xs), torch.stack(ids)

    return wrapped


batched_prc_img = _batched(prc_img)
batched_prc_vid = _batched(prc_vid)
batched_prc_audio = _batched(prc_audio)
batched_prc_txt = _batched(prc_txt)
batched_prc_txts = _batched(prc_txts)


def _compress_time(t_ids: Tensor) -> Tensor:
    t_max = torch.max(t_ids)
    remap = torch.zeros((t_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    uniq = torch.unique(t_ids, sorted=True)
    remap[uniq] = torch.arange(len(uniq), device=t_ids.device, dtype=t_ids.dtype)
    return remap[t_ids]


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    out_list = []
    for data, pos in zip(x, x_ids, strict=True):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_c = _compress_time(t_ids)
        t = int(torch.max(t_ids_c)) + 1
        h = int(torch.max(h_ids)) + 1
        w = int(torch.max(w_ids)) + 1

        flat_ids = t_ids_c * w * h + h_ids * w + w_ids
        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return out_list


def times_to_ids(time: Tensor) -> Tensor:
    return (time * 1000 // 10).to(dtype=torch.int64)


def times_to_ids_round(time: Tensor) -> Tensor:
    return torch.round(time * 100).to(dtype=torch.int64)
