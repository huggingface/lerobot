from __future__ import annotations
import argparse,json
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1/trials")
PLAN=Path("/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_act_additive_three_model_200k_v1/evaluation_plan.json")
def main():
 p=argparse.ArgumentParser();p.add_argument("--trial-id",required=True);p.add_argument("--success",action="store_true");p.add_argument("--failure-category");p.add_argument("--report-zh",required=True);a=p.parse_args()
 plan=json.loads(PLAN.read_text()); trial=next(t for t in plan["trials"] if t["trial_id"]==a.trial_id)
 out=ROOT/f"{a.trial_id}.operator_label.json"
 if out.exists(): raise FileExistsError(out)
 value={"schema":"task1_additive_three_model_eval24_operator_label_v1","evaluation_id":plan["evaluation_id"],
  "trial_id":a.trial_id,"model_key":trial["model_key"],"model_id":trial["model_id"],"created_at_utc":datetime.now(UTC).isoformat(),
  "status":"labeled","success":a.success,"failure_category":None if a.success else (a.failure_category or "operator_unspecified_failure"),
  "operator_report_zh":a.report_zh}
 out.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
 print(out)
if __name__=="__main__":main()
