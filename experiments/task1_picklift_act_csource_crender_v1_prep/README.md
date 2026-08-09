# Task1 C-source vs C-render preparation

Status: `offline_training_complete_eval24_software_gate_pass_hardware_not_authorized`.

The final rerender handoff was bound fail-closed, the 500-step smoke and fresh
200k run completed, and the fixed step-200000 checkpoint passed CUDA reload and
one Real/one rerender-Sim finite inference. `training_result_v1.json` is the
small committed result index; model/data/log artifacts remain outside Git.

The only intended training condition is C-render: frozen Real24 plus exactly the
same LocalSim-gap24 membership and state/action rows as the existing C-source
condition, with RGB replaced one-for-one by the accepted Real-like rerender.
The future binder validates this identity before creating a derived combined
Dataset or train configs.

The paired real evaluation is C-source versus C-render on the same frozen
24-pose Eval24 bank (48 trials, alternating SR/RS). Its plan and fake dry-run
are frozen, but `hardware_authorized=false`: no serial, camera, robot, torque,
or rollout was accessed during this work.
