# Task1 real-evaluation success early stop v1

This profile is only for future real evaluations that explicitly opt in. Existing
full-window plans and evidence remain unchanged.

The external producer atomically writes one JSON marker per trial at:

`<marker_root>/<evaluation_id>/<trial_id>.success.json`

Required payload fields are `evaluation_id`, `trial_id`,
`operator_confirmed_success=true`, and a timezone-aware UTC `created_at_utc`.
The operator sends it only after bilateral grasp, unsupported lift strictly over
5 cm, and a continuous hold of at least 0.5 seconds are visibly confirmed.

The evaluator checks the marker after recording and sending the current policy
tick. A valid marker ends the policy/video window with
`termination=success_early_stop`; return-to-ready and torque disable then follow
the existing out-of-window path. Missing, malformed, stale, future, or
wrong-identity markers are recorded as rejections and do not stop the trial.

Future paired plans opt in with this fragment (with the frozen profile SHA):

```json
{
  "success_early_stop": {
    "enabled": true,
    "explicit_opt_in": true,
    "profile_path": "experiments/task1_picklift_real24_act_v1/real_evaluation_success_early_stop_profile_v1.json",
    "profile_sha256": "50f4f2a2771cf21f34d333d13f2b413a54c3d183704218c1b0d36abef8e2aa28",
    "marker_root": "/home/ubuntu24/Teleop/artifacts/evaluation_success_markers"
  }
}
```

Do not add this block to a historical full-window plan. Independent video review
remains authoritative; a rejected early-stop signal scores failure without a
rerun.
