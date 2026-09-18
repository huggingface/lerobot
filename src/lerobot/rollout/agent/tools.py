# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Provider-neutral tool descriptions shared by the local server and API supervisor."""


def tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "strict": False,
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }


STRING = {"type": "string"}
NUMBER = {"type": "number"}
TARGETS = {"type": "object", "additionalProperties": NUMBER}
CONTROL_TOOLS = [
    tool("get_observation", "Get the latest camera views, named joints, proposal and control revision."),
    tool("get_action", "Inspect the latest VLA proposal; this does not execute or advance policy state."),
    tool(
        "set_task",
        "Set a new overall task and initial VLA instruction.",
        {"instruction": STRING},
        ["instruction"],
    ),
    tool(
        "steer",
        "Replace the active VLA instruction and invalidate old queued/in-flight actions.",
        {"instruction": STRING},
        ["instruction"],
    ),
    tool("pause", "Pause locally and discard queued actions."),
    tool("resume_policy", "Hand control back to the VLA after a tool motion."),
    tool(
        "move_joints",
        "Move named joints to absolute degree targets over a bounded duration, then hold.",
        {"targets": TARGETS, "duration_s": NUMBER, "instruction": STRING},
        ["targets", "duration_s"],
    ),
    tool(
        "offset_joints",
        "Apply degree offsets to currently observed joints, then hold.",
        {"targets": TARGETS, "duration_s": NUMBER, "instruction": STRING},
        ["targets", "duration_s"],
    ),
    tool(
        "set_gripper",
        "Set a named gripper joint in degrees, then hold.",
        {"joint": STRING, "position_deg": NUMBER, "duration_s": NUMBER},
        ["joint", "position_deg", "duration_s"],
    ),
    tool(
        "move_ee",
        "Move an arm using configured IK; pose is a 4x4 rigid transform with metres translation.",
        {
            "arm": STRING,
            "reference_frame": STRING,
            "pose": {"type": "array", "items": {"type": "array", "items": NUMBER}},
            "duration_s": NUMBER,
        },
        ["arm", "reference_frame", "pose", "duration_s"],
    ),
    tool(
        "finish_episode",
        "Save the attempt with outcome evidence; wait for an operator reset before the next.",
        {"outcome": {"type": "string", "enum": ["success", "failure", "unknown"]}, "evidence": STRING},
        ["outcome", "evidence"],
    ),
]
EXPERIMENT_TOOLS = [
    tool("list_jobs", "List experiment jobs and their statuses."),
    tool(
        "read_code",
        "Read a bounded range of a source file in the configured workspace.",
        {"path": STRING, "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}},
        ["path"],
    ),
    tool(
        "get_run_trace",
        "Read the last events from a recorded run for failure analysis.",
        {"run_id": STRING, "max_events": {"type": "integer"}},
        ["run_id"],
    ),
    tool("status", "Read active session, policy catalog and experiment status."),
    tool("list_candidates", "List available versioned training candidates."),
    tool(
        "create_candidate",
        "Create a training candidate. training_json is a LeRobot config; recipe_json is a TrainingRecipe.",
        {
            "name": STRING,
            "base_model": STRING,
            "training_json": STRING,
            "recipe_json": STRING,
            "parent": STRING,
        },
        ["name", "base_model", "training_json", "recipe_json"],
    ),
    tool(
        "start_training",
        "Launch fine-tuning using a saved candidate and LeRobot's training CLI.",
        {"candidate_id": STRING},
        ["candidate_id"],
    ),
    tool(
        "job_status",
        "Read status and the log tail of a training, data, or test job.",
        {"job_id": STRING},
        ["job_id"],
    ),
    tool("cancel_job", "Terminate a job launched by this service.", {"job_id": STRING}, ["job_id"]),
    tool(
        "register_policy",
        "Register an existing checkpoint for future rollout sessions; validates its config.",
        {"name": STRING, "checkpoint": STRING},
        ["name", "checkpoint"],
    ),
    tool(
        "start_rollout",
        "Load a registered checkpoint with the configured real robot and start a collection session.",
        {"policy_id": STRING, "instruction": STRING},
        ["policy_id", "instruction"],
    ),
    tool("stop_rollout", "Stop recording, finalize data and disconnect the robot."),
    tool(
        "build_dagger_dataset",
        "Create a new training dataset from seed demonstrations and selected correction episodes. spec_json supplies source roots and explicitly selected episode IDs.",
        {"spec_json": STRING},
        ["spec_json"],
    ),
    tool(
        "compare_runs",
        "Compare outcome and intervention metrics from two recorded runs.",
        {"baseline_run": STRING, "candidate_run": STRING},
        ["baseline_run", "candidate_run"],
    ),
    tool(
        "create_code_candidate",
        "Apply a proposed git diff in a separate worktree for a future session.",
        {"name": STRING, "patch": STRING},
        ["name", "patch"],
    ),
    tool(
        "test_code_candidate",
        "Run runtime and recipe tests in a code candidate worktree.",
        {"name": STRING},
        ["name"],
    ),
]
TOOLS = CONTROL_TOOLS + EXPERIMENT_TOOLS
