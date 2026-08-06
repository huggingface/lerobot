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
    """Derive the ulnar-side tendon command for a HopeJR finger from its glove-sensed MCP angles.

    The HopeJR hand flexes a finger with a pair of opposing tendons (radial and ulnar) rather than
    independent flexion and abduction joints. This blends the glove's flexion and abduction readings for
    one MCP joint into the ulnar tendon's share of the motion: an abduction toward the ulnar side pulls
    this tendon further, while `splay` sets how much of the abduction reading leaks into it versus pure
    flexion.

    Args:
        flexion (`float`):
            MCP flexion reading for the finger, as reported by the glove.
        abduction (`float`):
            MCP abduction reading for the finger, as reported by the glove. Positive values pull toward
            the radial side and are subtracted here.
        splay (`float`):
            Fraction, in `[0, 1]`, of the tendon command driven by abduction rather than flexion.

    Returns:
        `float`: The ulnar tendon's target position.
    """
    return -abduction * splay + flexion * (1 - splay)


def get_radial_flexion(flexion: float, abduction: float, splay: float):
    """Derive the radial-side tendon command for a HopeJR finger from its glove-sensed MCP angles.

    The counterpart to [`get_ulnar_flexion`]: same blend of flexion and abduction, but abduction toward
    the radial side adds to this tendon's target instead of subtracting from it.

    Args:
        flexion (`float`):
            MCP flexion reading for the finger, as reported by the glove.
        abduction (`float`):
            MCP abduction reading for the finger, as reported by the glove. Positive values pull toward
            the radial side and are added here.
        splay (`float`):
            Fraction, in `[0, 1]`, of the tendon command driven by abduction rather than flexion.

    Returns:
        `float`: The radial tendon's target position.
    """
    return abduction * splay + flexion * (1 - splay)


def homunculus_glove_to_hope_jr_hand(glove_action: dict[str, float]) -> dict[str, float]:
    """Translate a Homunculus Glove action into a HopeJR hand action.

    The glove reports one flexion and one abduction value per finger's MCP joint, plus a DIP/PIP reading,
    while the HopeJR hand is driven by a pair of tendons (radial and ulnar flexors) per finger and a
    coupled PIP/DIP joint. This remaps and blends the glove's per-joint keys into the hand's per-tendon
    keys via [`get_radial_flexion`] and [`get_ulnar_flexion`]; the thumb, whose joints map one-to-one, is
    passed through unchanged.

    Args:
        glove_action (`dict[str, float]`):
            Action produced by [`~teleoperators.homunculus.HomunculusGlove.get_action`], keyed by glove
            joint name.

    Returns:
        `dict[str, float]`: The equivalent action keyed by HopeJR hand joint name.
    """
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
