from __future__ import annotations
import hashlib,json,collections
from datetime import UTC,datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLAN=HERE/"evaluation_plan.json"
ROOT=Path("/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_repeat_realgap24_localsim24gap_act200k_eval24_v1")
TRIALS=ROOT/"trials"; OUT=ROOT/"operator_result_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 plan=json.loads(PLAN.read_text()); rows=[]
 for t in plan['trials']:
  ep=TRIALS/f"{t['artifact_stem']}.json"; lp=TRIALS/f"{t['artifact_stem']}.operator_label.json"
  if not ep.exists() or not lp.exists(): raise RuntimeError(f"missing {t['trial_id']}")
  e=json.loads(ep.read_text()); l=json.loads(lp.read_text())
  if e.get('run_error') is not None or e.get('torque_disable_verified') is not True: raise RuntimeError(t['trial_id'])
  rows.append({'trial_id':t['trial_id'],'pose_order':t['pose_order'],'model_key':t['model_key'],'model_id':t['model_id'],
   'cell':t['cell'],'coverage_tier':t['coverage_tier'],'yaw':t['nominal_yaw_degrees_modulo_90'],
   'operator_success':bool(l['success']),'failure_category':l.get('failure_category'),
   'evidence_sha256':sha(ep),'operator_label_sha256':sha(lp),'termination':e['termination'],
   'torque_disable_verified':True,'run_error':None})
 def group(field,value):
  x=[r for r in rows if r[field]==value];return {'success':sum(r['operator_success'] for r in x),'total':len(x),'rate':sum(r['operator_success'] for r in x)/len(x)}
 by_model={k:group('model_key',k) for k in 'ABC'}
 by_tier={k:group('coverage_tier',k) for k in sorted({r['coverage_tier'] for r in rows})}
 by_yaw={str(k):group('yaw',k) for k in (0,45)}
 by_cell={k:group('cell',k) for k in sorted({r['cell'] for r in rows})}
 summary={'schema':'task1_additive_three_model_eval24_operator_result_v1','status':'operator_labels_frozen_pending_canonical_video_review',
  'evaluation_id':plan['evaluation_id'],'plan_sha256':sha(PLAN),'trials':72,'operator_labels':72,
  'overall':{'success':sum(r['operator_success'] for r in rows),'total':72,'rate':sum(r['operator_success'] for r in rows)/72},
  'by_model':by_model,'by_coverage_tier':by_tier,'by_yaw':by_yaw,'by_cell':by_cell,
  'integrity':{'run_error_null':72,'torque_disable_verified':72,'maximum_duration':sum(r['termination']=='maximum_duration' for r in rows),
   'success_early_stop':sum(r['termination']=='success_early_stop' for r in rows)},
  'claim_boundary':'Operator-stage engineering result; canonical-video review pending; no causal paper conclusion.',
  'created_at_utc':datetime.now(UTC).isoformat()}
 OUT.mkdir(parents=True)
 with (OUT/'trials.jsonl').open('x') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n')
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 files=[OUT/'trials.jsonl',OUT/'summary.json',PLAN]
 (OUT/'hashes.sha256').write_text(''.join(f"{sha(p)}  {p}\n" for p in files))
 (HERE/'operator_result_index.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
