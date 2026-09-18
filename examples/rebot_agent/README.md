# ReBot physical agent experiment

Start with [the physical agent guide](../../docs/source/physical_agent_loop.mdx).
`session.json` pins the annotated dataset and defines SmolVLA/Pi0.5 candidates.
`steerable_80_20.yaml` trains actions on subtask instructions (80%) and overall tasks
(20%) through LeRobot's existing weighted recipe renderer.

Run `lerobot-rollout --agent_config=examples/rebot_agent/session.json` on the host
that will train or control the robot. Training can start before any checkpoint exists.
Before a physical rollout, replace the arm/camera placeholders and register a trained
checkpoint. The checked-in file does not contain installation calibration or credentials.

[astra_goal.md](astra_goal.md) is the reusable system/goal prompt for the complete
physical improvement loop. Paste it into an external agent's goal, or configure the
built-in live supervisor with:

```json
"supervisor": {
  "model": "gpt-6-astra",
  "prompt_path": "examples/rebot_agent/astra_goal.md",
  "timeout_s": 30
}
```

The built-in supervisor polls active robot sessions. An external experiment agent
uses the same prompt and tool API to drive training and improvement between sessions.
