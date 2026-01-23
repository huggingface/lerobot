# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

INDEX_SPLAY = 0.3
MIDDLE_SPLAY = 0.3
RING_SPLAY = 0.3
PINKY_SPLAY = 0.5


def get_ulnar_flexion(flexion: float, abduction: float, splay: float):
    return -abduction * splay + flexion * (1 - splay)


def get_radial_flexion(flexion: float, abduction: float, splay: float):
    return abduction * splay + flexion * (1 - splay)


def homunculus_glove_to_hope_jr_hand(glove_action: dict[str, float]) -> dict[str, float]:
    return {
        "thumb_cmc.pos": glove_action["thumb_cmc.pos"],
        "thumb_mcp.pos": glove_action["thumb_mcp.pos"],
        "thumb_pip.pos": glove_action["thumb_pip.pos"],
        "thumb_dip.pos": glove_action["thumb_dip.pos"],
        "index_radial_flexor.pos": get_radial_flexion(
            glove_action["index_mcp_flexion.pos"], glove_action["index_mcp_abduction.pos"], INDEX_SPLAY
        ),
        "index_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["index_mcp_flexion.pos"], glove_action["index_mcp_abduction.pos"], INDEX_SPLAY
        ),
        "index_pip_dip.pos": glove_action["index_dip.pos"],
        "middle_radial_flexor.pos": get_radial_flexion(
            glove_action["middle_mcp_flexion.pos"], glove_action["middle_mcp_abduction.pos"], MIDDLE_SPLAY
        ),
        "middle_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["middle_mcp_flexion.pos"], glove_action["middle_mcp_abduction.pos"], MIDDLE_SPLAY
        ),
        "middle_pip_dip.pos": glove_action["middle_dip.pos"],
        "ring_radial_flexor.pos": get_radial_flexion(
            glove_action["ring_mcp_flexion.pos"], glove_action["ring_mcp_abduction.pos"], RING_SPLAY
        ),
        "ring_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["ring_mcp_flexion.pos"], glove_action["ring_mcp_abduction.pos"], RING_SPLAY
        ),
        "ring_pip_dip.pos": glove_action["ring_dip.pos"],
        "pinky_radial_flexor.pos": get_radial_flexion(
            glove_action["pinky_mcp_flexion.pos"], glove_action["pinky_mcp_abduction.pos"], PINKY_SPLAY
        ),
        "pinky_ulnar_flexor.pos": get_ulnar_flexion(
            glove_action["pinky_mcp_flexion.pos"], glove_action["pinky_mcp_abduction.pos"], PINKY_SPLAY
        ),
        "pinky_pip_dip.pos": glove_action["pinky_dip.pos"],
    }
