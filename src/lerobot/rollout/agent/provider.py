# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Optional Responses API supervisor; the local tool API is usable by any agent/provider."""

import base64
import io
import json
import os

import numpy as np
import requests
from PIL import Image

DEFAULT_PROMPT = """You supervise a real robot and improve a language-steerable VLA.
The VLA normally produces the actions. Inspect current camera views, instructions,
recent tool outcomes and action proposals. Use steer to clarify the next instruction;
use bounded motion tools to correct a mistake; explicitly resume_policy after recovery.
A tool reports execution, not task success. Check observations before declaring success.
Never infer calibrated metric positions from pixels alone. Use only the supplied joint
names, limits, coordinate frames and tools. Respond with at most one tool call per turn.
During an active collection session prioritize progress and supervision. Between sessions
you may create training candidates, launch fine-tuning, assemble reviewed correction data,
and compare experiments. Never label a rollout successful merely to improve the metric.
Keep evaluation tasks and scoring fixed. Code candidates run in separate worktrees and
must pass tests before use in a new session. Operator pauses and episode reset waits are
explicit boundaries; only the operator starts/resumes those sessions.
"""


def image_data_url(array):
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("Supervisor camera views must be HWC uint8 RGB")
    img = Image.fromarray(array)
    img.thumbnail((640, 480))
    stream = io.BytesIO()
    img.save(stream, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def public_snapshot(snapshot, *, images=True):
    if snapshot is None:
        return None
    output = {key: value for key, value in snapshot.items() if key != "raw"}
    output["images"] = {
        key: image_data_url(value)
        for key, value in snapshot["raw"].items()
        if isinstance(value, np.ndarray) and value.ndim == 3 and images
    }
    return output


class OpenAIReasoner:
    def __init__(
        self,
        *,
        model="gpt-6-astra",
        prompt=None,
        timeout_s=30.0,
        key_env="OPENAI_API_KEY",
        reasoning_effort="low",
    ):
        self.model, self.prompt, self.timeout_s = model, prompt or DEFAULT_PROMPT, timeout_s
        self.reasoning_effort = reasoning_effort
        self.key = os.environ.get(key_env)
        if not self.key:
            raise ValueError(f"Set {key_env} to enable the API supervisor, or use the local tool API")

    def decide(self, snapshot, tools, context):
        observation = public_snapshot(snapshot)
        images = observation.pop("images", {}) if observation else {}
        content = [{"type": "input_text", "text": json.dumps({"observation": observation, **context})}]
        for camera, url in images.items():
            content.extend(
                [
                    {"type": "input_text", "text": f"Camera: {camera}"},
                    {"type": "input_image", "image_url": url},
                ]
            )
        response = requests.post(
            "https://api.openai.com/v1/responses",
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "instructions": self.prompt,
                "input": [{"role": "user", "content": content}],
                "tools": tools,
                "parallel_tool_calls": False,
                "reasoning": {"effort": self.reasoning_effort},
                "store": False,
            },
        )
        response.raise_for_status()
        calls = [
            {"name": item["name"], "arguments": json.loads(item["arguments"])}
            for item in response.json().get("output", [])
            if item["type"] == "function_call"
        ]
        if len(calls) > 1:
            raise ValueError("The physical supervisor must return at most one tool call per decision")
        return calls
