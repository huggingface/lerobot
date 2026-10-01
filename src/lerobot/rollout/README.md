# Direct visual control and hybrid supervision

The optional external planner extends Pablo’s steering interface. `hybrid.vlm_only=true`
selects direct VLM control: omit all `--policy.*` flags and use `--inference.type=sync`.
No VLA checkpoint, inference worker, tokenizer or normalization statistics are loaded.
A planner configuration and a complete `hybrid.limits` / `hybrid.end_effectors` contract
are still required; keep those hardware-specific values in the rig configuration.

In VLM-only mode, each cycle sends current images, measured native positions, FK,
camera-mount provenance and the preceding motion result. The VLM returns exactly one
JSON tool request: `move_to`, `move_gripper`, `look`, `done`, or `give_up`, with `scene`,
`note`, `ee_targets` and `targets`. The controller owns the move duration; the model
cannot choose speeds, alter limits or execute code. One end effector may move per
request, with optional gripper targets. Local bounded IK resolves the absolute tool
pose to joints, then the ordinary robot processor/driver sends commands. Unmentioned
axes hold their targets. Freshness, range, speed, IK and reply-epoch checks remain active.

After motion, measured joint/FK arrival and residuals accompany the next observation.
A rejected target is never executed and its error is returned for revision. Failure
to reach a valid target becomes feedback; the controller holds the measured pose
instead of repeatedly forcing the old endpoint. Hardware errors and unavailable API
transport still stop the segment. `done` requests normal completion/optional home;
`give_up` requests operator attention. There is no artificial consecutive-move cap
in this mode. Interactive `/start`, `/subtask`, `/stop`, `/reset` and `/quit` still apply;
manual goal changes invalidate outstanding replies and remain under direct VLM control.

This follows [inspect-robots-agent](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent)
in its one-motion/observation loop, explanatory action notes and measured feedback.
It is an original LeRobot implementation using the shared JSON planner transport,
not the upstream native function-tool protocol, evaluator, hindsight-learning system
or I2RT controller. No collision planner, depth localization or calibrated object
coordinates are implied by FK or an estimated camera mount. Joint interpolation is
not a straight Cartesian path. The existing hybrid mode still uses Molmo/VLA proposals;
VLM-only mode never delegates motion to a VLA.
