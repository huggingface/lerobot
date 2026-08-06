from __future__ import annotations
import hashlib,json
from datetime import UTC,datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLAN=HERE/"evaluation_plan.json"
RESULT=HERE/"training_result_v1.json"
DRY=Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1/software_preparation_v1/dry_run.json")
OUT=DRY.parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def main():
 plan=json.loads(PLAN.read_text()); dry=json.loads(DRY.read_text()); result=json.loads(RESULT.read_text())
 assert len(plan["trials"])==72 and dry["status"]=="software_dry_run_pass_hardware_not_accessed"
 assert result["status"]=="three_models_offline_training_complete"
 poses=[plan["trials"][i] for i in range(0,72,3)]
 assert len({t["cell"] for t in poses})==12
 assert set({cell:sum(t["cell"]==cell for t in poses) for cell in {t["cell"] for t in poses}}.values())=={2}
 manifest={"schema":"task1_additive_three_model_eval24_software_gate_v1","status":"pass_hardware_not_authorized",
  "evaluation_id":plan["evaluation_id"],"plan":{"path":str(PLAN),"sha256":sha(PLAN)},
  "training_result":{"path":str(RESULT),"sha256":sha(RESULT)},"dry_run":{"path":str(DRY),"sha256":sha(DRY)},
  "trial_count":72,"pose_count":24,"models":{k:v["model_sha256"] for k,v in plan["models"].items()},
  "balance":plan["balance"],"model_order":plan["model_order"],"success_early_stop":plan["success_early_stop"],
  "hardware_access":{"serial":False,"camera":False,"robot":False,"torque":False,"rollout":False},
  "first_trial":plan["trials"][0],"created_at_utc":datetime.now(UTC).isoformat()}
 (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 files=[PLAN,RESULT,DRY,OUT/"manifest.json"]
 (OUT/"hashes.sha256").write_text("".join(f"{sha(p)}  {p}\n" for p in files))
 (HERE/"software_gate_result_index.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__":main()
