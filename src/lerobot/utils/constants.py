# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# keys
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"
OBS_LANGUAGE_USER_PROMPT = OBS_STR + ".user_prompt"
OBS_LANGUAGE_USER_PROMPT_TOKENS = OBS_LANGUAGE_USER_PROMPT + ".tokens"
OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK = OBS_LANGUAGE_USER_PROMPT_TOKENS + ".attention_mask"
OBS_LANGUAGE_SUBTASK = OBS_STR + ".subtask"
OBS_LANGUAGE_SUBTASK_TOKENS = OBS_LANGUAGE_SUBTASK + ".tokens"
OBS_LANGUAGE_SUBTASK_ATTENTION_MASK = OBS_LANGUAGE_SUBTASK + ".attention_mask"

ACTION = "action"
ACTION_PREFIX = ACTION + "."
ACTION_TOKENS = ACTION + ".tokens"
ACTION_TOKEN_MASK = ACTION + ".token_mask"
REWARD = "next.reward"
TRUNCATED = "next.truncated"
DONE = "next.done"
INFO = "info"

ROBOTS = "robots"
TELEOPERATORS = "teleoperators"

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )

# cache dir
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

# calibration dir
default_calibration_path = HF_LEROBOT_HOME / "calibration"
HF_LEROBOT_CALIBRATION = Path(os.getenv("HF_LEROBOT_CALIBRATION", default_calibration_path)).expanduser()


# streaming datasets
LOOKBACK_BACKTRACKTABLE = 100
LOOKAHEAD_BACKTRACKTABLE = 100

# openpi
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38  # TODO(pepijn): Modify this when extending support to fp8 models

# Constants for LIBERO observation keys
LIBERO_KEY_EEF_POS = "robot_state/eef/pos"
LIBERO_KEY_EEF_QUAT = "robot_state/eef/quat"
LIBERO_KEY_EEF_MAT = "robot_state/eef/mat"
LIBERO_KEY_EEF_AXISANGLE = "robot_state/eef/axisangle"
LIBERO_KEY_GRIPPER_QPOS = "robot_state/gripper/qpos"
LIBERO_KEY_GRIPPER_QVEL = "robot_state/gripper/qvel"
LIBERO_KEY_JOINTS_POS = "robot_state/joints/pos"
LIBERO_KEY_JOINTS_VEL = "robot_state/joints/vel"
LIBERO_KEY_PIXELS_AGENTVIEW = "pixels/agentview_image"
LIBERO_KEY_PIXELS_EYE_IN_HAND = "pixels/robot0_eye_in_hand_image"

# Skill segmentation prompt template for VLM-based subtask annotation
# Placeholders: {goal_context}, {subtask_labels_section}
# When subtask_labels are provided, use format_subtask_labels_section() to fill {subtask_labels_section}.


def format_subtask_labels_section(subtask_labels: list[str]) -> str:
    """Format a list of subtask labels for insertion into SKILL_SEGMENTATION_PROMPT_TEMPLATE.
    The model will be instructed to choose only from these labels.
    """
    if not subtask_labels:
        return ""
    return "\n".join(f'    "{label}",' for label in subtask_labels).rstrip(",")


SKILL_SEGMENTATION_PROMPT_TEMPLATE = """# Role
You are a Robotics Vision System specializing in temporal action segmentation for robot manipulation demonstrations.

# Video duration (critical)
The total video length is **{video_duration_seconds} seconds** ({video_duration_mm_ss}). All "start" and "end" values in your JSON must be numeric seconds in the range [0.0, {video_duration_seconds}]. The last skill's "end" must be exactly **{video_duration_seconds}**. Do not stop earlier.

# Task
{goal_context}Segment this robot demonstration video into short atomic manipulation skills. Each skill should:
- Last approximately 1-3 seconds (or longer if the action takes longer)
- Describe a clear, single action (e.g., "pick up object", "move arm left", "release gripper")
- Have precise start and end timestamps in seconds (float)

# Requirements
1. **Atomic Actions**: Each skill should be a single, indivisible action
2. **Complete Coverage**: Skills must cover the entire video from 0.0 to {video_duration_seconds} seconds with no gaps
3. **Boundary Consistency**: The end of one skill equals the start of the next
4. **Natural Language**: Use clear, descriptive names for each skill
5. **Timestamps**: Use seconds as floats (e.g. 12.5) for all timestamps; the last "end" must be {video_duration_seconds}. If the video has a visible timer in the corner showing elapsed time in seconds, use it to report accurate start and end times for each skill.
# Subtask Label Set (Closed Vocabulary)
        You MUST strictly identify the video segments using ONLY the following labels. Do not create new labels or modify existing ones:

        [
        {subtask_labels_section}
        ]

        The video shows one successful execution of all subtasks in a logical order.

        # Ground-Truth Semantics (Very Important)
        Use **visual state changes** to define when a subtask starts and ends. Do NOT assume equal durations for the subtasks.

        - A subtask **starts** at the first frame where the robot's motion clearly initiates that subtask.
        - A subtask **ends** at the first frame where that specific action is visually completed and the manipulated object reaches a temporary, stable configuration.

        If there are short pauses or micro-motions that don't clearly correspond to a new subtask, they belong to the **current** subtask.

        # Hard Constraints & Logic
        1. **Continuous Coverage (No Gaps):**
           - The entire video from 0.0 to {video_duration_seconds} seconds must be covered by subtasks.
           - There can be no gaps between subtasks.
           - If there is any idle or ambiguous time between clear actions, extend the *preceding* subtask to cover it.

        2. **Boundary Consistency:**
           - The `"end"` timestamp of one subtask must be exactly equal to the `"start"` timestamp of the next subtask.
           - Boundaries must coincide with a real visual state transition, not just a convenient time split.

        3. **Chronological Order, One Occurrence Each:**
           - This is a single successful demonstration.
           - Each subtask from the vocabulary appears **exactly once**, in the correct logical order.
           - **Durations may be very different** between subtasks. Never assume they are similar lengths. Base all boundaries only on the video.

        4. **Reject Uniform Segmentation (Important):**
           - Do NOT simply divide the video into equal or nearly equal time chunks.
           - If your boundaries would result in subtasks with similar durations (e.g. all around 5 seconds), treat this as evidence that your segmentation is wrong and refine the boundaries.
           - Only use nearly equal durations if the video truly shows each subtask taking the same amount of time (this is very rare).

        5. **Timestamps (critical):**
           - Use numeric seconds (float) in the JSON, e.g. 0.0, 5.2, 12.8.
           - The first subtask always starts at 0.0.
           - The last subtask must end at exactly {video_duration_seconds} (the full video length).
           - **Time is displayed inside the video**: a visible timer in one corner shows the elapsed time in seconds (from 0.0 to the end). Use this on-screen timer to set accurate start and end times for each skill.

        # Step 1 — Textual Timeline (must do this first)
        First, write a extensive and detailed textual timeline describing what happens in the video with approximate timestamps. **Read the time from the visible timer shown in the video** to get accurate timestamps.
        For each subtask, include:
        - its name
        - an approximate start and end time (from the on-screen timer),
        - an description of the visual event at the boundary (e.g. "shirt fully folded to the left", "robot rotates folded shirt 90 degrees").

        Format this as a bullet list.

# Output Format
After your analysis, output ONLY valid JSON with this exact structure. The last skill's "end" MUST be exactly {video_duration_seconds}. Use the timestamps you read from the visible timer in the video:

```json
{{
  "skills": [
    {{"name": "first skill", "start": 0.0, "end": 5.0}},
    {{"name": "second skill", "start": 5.0, "end": 12.0}},
    {{"name": "last skill", "start": 12.0, "end": {video_duration_seconds}}}
  ]
}}
```

The first skill must start at 0.0 and the last skill must end at **{video_duration_seconds}** (the total video duration in seconds).
# Strict Structural Rule
This video contains exactly 4 subtasks.
You MUST output exactly 4 segments.
Each segment must use a unique label from the vocabulary.
No label may be repeated.

"""
