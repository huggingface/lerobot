from __future__ import annotations

import hashlib, json, re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

HERE = Path(__file__).resolve().parent
ART = Path("/home/ubuntu24/Teleop/artifacts")
EXP = "task1_picklift_act_additive_three_model_200k_v1"
TRAIN = ART / f"training/{EXP}"
EVIDENCE = ART / f"evidence/{EXP}/training_result_v1"
CONDITIONS = {
    "A": ("R24_repeat_200k", "r24_repeat", Path("/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real24_budget_extension_v1/accepted"), "local/task1_picklift_real24_budget_extension_v1_accepted", [(0,"real24_stream_a"),(1,"real24_stream_b")]),
    "B": ("R24_real_gap24_200k", "r24_realgap24", ART / f"datasets/{EXP}/real24_realgap24/combined48_v1", "local/task1_picklift_real24_realgap24_additive_v1", [(0,"real24"),(4263,"real_gap24")]),
    "C": ("R24_localsim_gap24_200k", "r24_localsim24gap", ART / f"datasets/{EXP}/real24_localsim24gap/combined48_v1", "local/task1_picklift_real24_localsim24gap_additive_v1", [(0,"real24"),(4263,"localsim_gap24")]),
}

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def write(p: Path, x: object): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")

def metric(log: Path) -> dict:
    lines=log.read_text(errors="replace").replace("\r","\n").splitlines()
    line=[x for x in lines if re.search(r"ot_train\.py:\d+ step:",x)][-1]
    fields=dict(re.findall(r"([A-Za-z0-9_/]+):(\S+)",line))
    return {k: fields[k] for k in ("step","loss","l1_loss","kld_loss","grdn","smp/s")}

def validate(key: str, name: str, root: Path, repo: str, samples: list[tuple[int,str]]) -> dict:
    ckpt=TRAIN/name/"full_200k/checkpoints/200000/pretrained_model"
    ds=LeRobotDataset(repo,root=root,video_backend="pyav")
    model=ACTPolicy.from_pretrained(ckpt).to("cuda").eval()
    pre,post=make_pre_post_processors(policy_cfg=model.config,pretrained_path=str(ckpt),preprocessor_overrides={"device_processor":{"device":"cuda"}})
    rows=[]
    for idx,label in samples:
        s=ds[idx]; inputs={"observation.state":s["observation.state"].unsqueeze(0),"observation.images.front":s["observation.images.front"].unsqueeze(0)}
        model.reset()
        with torch.inference_mode(): out=post(model.select_action(pre(inputs))).cpu().numpy()
        if out.shape!=(1,6) or not np.isfinite(out).all(): raise RuntimeError(f"{key} invalid output")
        rows.append({"domain":label,"index":idx,"output_shape":[1,6],"finite":True,"action":out[0].tolist()})
    stats=ckpt/"policy_preprocessor_step_3_normalizer_processor.safetensors"; tensors=load_file(stats)
    if not np.allclose(tensors["observation.images.front.mean"].numpy().reshape(-1),[.485,.456,.406],atol=1e-7): raise RuntimeError("ImageNet stats mismatch")
    return {"status":"pass","checkpoint":str(ckpt),"model_sha256":sha(ckpt/"model.safetensors"),
            "config_sha256":sha(ckpt/"config.json"),"train_config_sha256":sha(ckpt/"train_config.json"),
            "policy_preprocessor_sha256":sha(ckpt/"policy_preprocessor.json"),"processor_stats_sha256":sha(stats),
            "samples":rows,"cuda_reload":True,"hardware_accessed":False}

def main():
    if EVIDENCE.exists(): raise FileExistsError(EVIDENCE)
    models={}; details={}
    for key,(model_id,name,root,repo,samples) in CONDITIONS.items():
        log=ART/f"evidence/{EXP}/formal_logs/{name}.log"
        if "End of training" not in log.read_text(errors="replace"): raise RuntimeError(f"{name} incomplete")
        count_path=TRAIN/name/"full_200k/matched_two_stream_sampling_counts.json"; counts=json.loads(count_path.read_text())
        if counts["actual_samples_seen_by_main_process"]!={"real24":800000,"source_b":800000}: raise RuntimeError(f"{name} count mismatch")
        checkpoints={str(step):sha(TRAIN/name/f"full_200k/checkpoints/{step:06d}/pretrained_model/model.safetensors") for step in (20000,40000,60000,80000,100000,120000,140000,160000,180000,200000)}
        validation=validate(key,name,root,repo,samples); write(EVIDENCE/f"offline_validation_{key}.json",validation)
        model={"model_id":model_id,"checkpoint":validation["checkpoint"],**{k:validation[k] for k in ("model_sha256","config_sha256","train_config_sha256","policy_preprocessor_sha256","processor_stats_sha256")}}
        models[key]=model; details[key]={"metrics":metric(log),"checkpoints":checkpoints,"sampling_counts":counts,"log_sha256":sha(log),"offline_validation":validation}
    result={"schema":"task1_additive_three_model_act200k_result_v1","status":"three_models_offline_training_complete",
            "selected_step":200000,"models":models,"details":details,"completed_at_utc":datetime.now(UTC).isoformat(),
            "boundaries":{"hardware_accessed":False,"rollout_started":False,"paper_effect_claim":False}}
    write(HERE/"training_result_v1.json",result); write(EVIDENCE/"training_result_v1.json",result)
    files=[HERE/"training_result_v1.json",*(EVIDENCE/f"offline_validation_{k}.json" for k in "ABC")]
    (EVIDENCE/"hashes.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__": main()
