# port plan: reward models lerobot-panda -> lerobot (VAlikV fork)

branch: feature/reward-models-port
src: /home/dom-iva/github.com/orel/lerobot/lerobot-panda (feature/hil-serl-updated)
dst: /home/dom-iva/github.com/orel/lerobot/lerobot
note: src uses hydra. skip hydra. use fork's JSON env cfg pattern + dataclass cfgs.

## fork state (already present)

- processor/hil_processor.py :: RewardClassifierProcessorStep (CNN path baked-in, no dispatch)
- processor/hil_processor.py :: InterventionActionProcessorStep (manual success = teleop SUCCESS btn)
- policies/sac/reward_model/ (modeling_classifier.py, configuration_classifier.py, processor_classifier.py)
- policies/sarm/ (modeling_sarm.py, configuration_sarm.py, processor_sarm.py=train-side, sarm_utils.py, compute_rabc_weights.py)
- rl/gym_manipulator.py :: gym_hil wiring, `cfg.name=="gym_hil"` -> PandaPickCubeGamepad-v0
- rl/*.json :: env configs (no hydra)
- hilserl extra :: `pip install lerobot[hilserl]` pulls gym-hil>=0.1.13

## gaps (what to port)

### A. unified reward processor hub
target: src/lerobot/processor/reward_model/
files:
- `__init__.py` <- src __init__ (imports + patch apply)
- `base.py` <- src base.py (RewardModelConfig + BaseRewardProcessorStep abstract)
- `_classifier_patch.py` <- src _classifier_patch.py (monkey-patch Classifier.__init__ & predict_reward to use classifier_preprocessor pipeline)
- `height_gripper.py` <- src (manual heuristic: EE z + gripper closed); add to fork
- `cnn_classifier.py` <- src (wraps upstream Classifier, loads via from_pretrained, calls predict_reward w/ threshold)
- `sarm.py` <- src (ring-buffer past frames; replicate current into future slots; reward_mode=binary|dense|delta; stats_dataset_repo_id)

port note:
- swap imports: `lerobot_panda.*` -> `lerobot.*`
- swap `lerobot_panda.processor.reward_model.*` -> `lerobot.processor.reward_model.*`
- sarm processor references SARMConfig / SARMRewardModel / sarm_utils -> remap to `lerobot.policies.sarm.*`
- register steps in `ProcessorStepRegistry` consistent with fork style

### B. dispatcher + gym_manipulator wiring
target: src/lerobot/rl/gym_manipulator.py
action: add `_build_reward_model_step(cfg)` dispatch on `cfg.type in {manual,cnn,sarm}`
- manual -> no processor (InterventionActionProcessorStep already provides success via teleop btn)
  - alternatively emit HeightGripperRewardStep when `cfg.type == height_gripper`
- cnn -> CNNRewardProcessorStep (new) OR keep existing RewardClassifierProcessorStep; prefer new (has patch)
- sarm -> SARMRewardProcessorStep
inject step AFTER TimeLimitProcessorStep, BEFORE AddBatchDimension (match src layout)

### C. cfg surface (no hydra)
use dataclasses + JSON env file keys (match existing rl/*.json pattern)
```json
"processor": {
  "terminate_on_success": true,
  "reward_model": {
    "type": "sarm",
    "pretrained_path": "outputs/lift_peg_sarm/checkpoints/last/pretrained_model",
    "device": "cuda",
    "task": "lift_cube",
    "reward_mode": "delta",
    "success_threshold": 0.9,
    "stats_dataset_repo_id": "lerobot/pusht_cropped_resized"
  }
}
```
parse into dataclass instances via simple dispatch by "type" key.

### D. training + data utils
target: src/lerobot/rl/
- `prepare_sarm_data.py` <- src (merge success + failure demos, split train/val; produces `{combined,combined-train,combined-val}` repo ids)
- `relabel_sarm.py` <- src (resample offline buffer with SARM rewards; must match reward_mode used online)

target: src/lerobot/policies/sac/reward_model/
- `split_dataset.py` (NEW, adapted from src/lerobot_panda/reward_classifier/split_dataset.py) :: frame-level train/val split
- note: stats must be preserved -> after split point stats_dataset_repo_id to source ds

target: src/lerobot/scripts/
- `lerobot_train_reward_classifier.py` (NEW) :: thin wrapper over lerobot_train.py that imports `lerobot.processor.reward_model._classifier_patch` BEFORE calling upstream train_cli() — ensures patch applied

### E. QoL (optional but useful)
- crop_dataset_roi.py already in fork rl/
- check: is there a train-wrapper that auto-applies classifier_patch on policy=classifier? if not, ensure `lerobot/processor/reward_model/__init__.py` is imported early (side-effect applies patch)

## files manifest

port (copy+edit imports):
```
src/lerobot_panda/processor/reward_model/__init__.py        -> src/lerobot/processor/reward_model/__init__.py
src/lerobot_panda/processor/reward_model/base.py            -> src/lerobot/processor/reward_model/base.py
src/lerobot_panda/processor/reward_model/_classifier_patch.py -> ...
src/lerobot_panda/processor/reward_model/height_gripper.py  -> ...
src/lerobot_panda/processor/reward_model/cnn_classifier.py  -> ...
src/lerobot_panda/processor/reward_model/sarm.py            -> ...
src/lerobot_panda/rl/prepare_sarm_data.py                   -> src/lerobot/rl/prepare_sarm_data.py
src/lerobot_panda/rl/relabel_sarm.py                        -> src/lerobot/rl/relabel_sarm.py
src/lerobot_panda/reward_classifier/split_dataset.py        -> src/lerobot/policies/sac/reward_model/split_dataset.py
src/lerobot_panda/scripts/train_reward_classifier.py (if exists) -> src/lerobot/scripts/lerobot_train_reward_classifier.py
```

modify (wire dispatch):
```
src/lerobot/rl/gym_manipulator.py     -- add _build_reward_model_step, pipeline insert
src/lerobot/envs/configs.py           -- extend HILSerlProcessorConfig with reward_model dict
src/lerobot/processor/__init__.py     -- re-export reward_model steps (optional, if convenient)
```

rl JSON env cfgs to add (examples):
```
src/lerobot/rl/gym_hil_manual_env.json    -- type=manual (or height_gripper) w/ z/gripper idx for gym_hil
src/lerobot/rl/gym_hil_cnn_env.json       -- type=cnn w/ pretrained_path
src/lerobot/rl/gym_hil_sarm_env.json      -- type=sarm w/ pretrained_path
```

## sim env mapping (gym_hil PandaPickCubeGamepad-v0)

- state dim: 18 (not 8). height_gripper indices must differ from panda hw defaults.
- image key: likely `observation.images.front` or similar — probe `env.reset()[0].keys()` once.
- action: 3d delta XYZ + gripper (wrapped from 7d base). matches gym_hil.
- reward: sparse (lift > 10 cm). success signal already present, but we'll override with our reward_model to test port.

actions before starting impl:
1. `uv pip install -e '.[hilserl]'` (or equivalent) to pull gym-hil
2. probe env state+image keys via small script

## risks

- R1: fork's existing RewardClassifierProcessorStep duplicates new CNN step — keep both for b/c OR replace; prefer replace w/ deprecation note in docstring
- R2: SARMEncodingProcessorStep (fork) vs SARMRewardProcessorStep (new) share zero code — ok, different roles (train vs inference)
- R3: `stats_dataset_repo_id` resolution: sarm processor loads stats from dataset meta. fork's DataProcessorPipeline API may differ slightly. verify in impl.
- R4: classifier_preprocessor.json schema may have drifted — check after first load attempt
- R5: gym_hil state layout differs from panda hw → height_gripper thresholds must retune or switch to sparse env reward during smoke

## non-goals

- hydra port (explicitly skipped per user)
- hw integration testing (sim only for port validation)
- real-robot configs (lerobot_panda robot/teleop layer is panda-specific, orthogonal to reward models)

## commit hygiene

- .beads/ not committed
- tests/ not committed
- docs/port/*.md not committed
- commit only src/ changes
- commits from domrachev03 identity, no co-author tag
