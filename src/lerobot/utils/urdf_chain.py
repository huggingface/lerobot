# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Minimal URDF parsing for a serial chain from base to a tip link (visualization)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET


def kinematic_chain_links(urdf_path: str, tip_link: str, base_link: str | None = None) -> list[str]:
    """Return ordered link names from base to tip along parent→child joints.

    Walks from ``tip_link`` backward via joint parents until ``base_link`` (or inferred root).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    parent_of: dict[str, str] = {}
    for j in root.findall("joint"):
        p_el = j.find("parent")
        c_el = j.find("child")
        if p_el is None or c_el is None:
            continue
        p = p_el.get("link")
        c = c_el.get("link")
        if p and c:
            parent_of[c] = p

    if tip_link not in parent_of and tip_link != base_link:
        # Tip might be root in weird URDFs; still return singleton
        if base_link:
            return [base_link, tip_link] if tip_link != base_link else [tip_link]
        return [tip_link]

    if base_link is None:
        all_parents = set(parent_of.values())
        all_children = set(parent_of.keys())
        roots = all_parents - all_children
        if roots:
            base_link = sorted(roots)[0]
        else:
            base_link = parent_of.get(tip_link, tip_link)

    chain_rev: list[str] = [tip_link]
    cur = tip_link
    seen: set[str] = set()
    while cur in parent_of and cur not in seen:
        seen.add(cur)
        par = parent_of[cur]
        chain_rev.append(par)
        if par == base_link:
            break
        cur = par
        if len(chain_rev) > 128:
            break

    chain = list(reversed(chain_rev))
    return chain if chain else [tip_link]


def robot_urdf_file_in_dir(urdf_dir: str) -> str:
    path = os.path.join(urdf_dir, "robot.urdf")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No robot.urdf in directory: {urdf_dir}")
    return path
