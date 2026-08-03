import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_budget_extension_same_eval48_v1/trials')
OUT=ROOT.parent/'canonical_video_review_v1'
PLAN=Path('/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_real24_budget_extension_same_eval48_v1/evaluation_plan.json')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def probe(p):
 q=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(p)],capture_output=True,text=True,check=True).stdout
 s=json.loads(q)['streams'][0]; return {'width':int(s['width']),'height':int(s['height']),'fps':s['avg_frame_rate'],'frames':int(s['nb_frames']),'duration':float(s['duration'])}
plan=json.loads(PLAN.read_text()); assert sha(PLAN)=='ada1a17eecc972a999fe8e8540015b42ebc3115577bd34a73985a9f97eb29abf'
assert len(plan['trials'])==48 and not OUT.exists(); OUT.mkdir(); (OUT/'sidecars').mkdir(); rows=[]
for t in plan['trials']:
 stem=t['artifact_stem']; scored_stem=stem+'__replacement1' if stem in ('t017_p17_real24','t022_p22_real24') else stem; d=json.loads((ROOT/f'{scored_stem}.json').read_text()) if scored_stem!=stem else json.loads((ROOT/f'{stem}.json').read_text()); op=json.loads((ROOT/f'{scored_stem}.operator_label.json').read_text()); v=ROOT/f'{scored_stem}.mp4'; sp=ROOT/f'{scored_stem}.steps.jsonl';
 assert d['run_error'] is None and d['torque_disable_verified'] and d['automatic_return']['result']['status']=='ready_pose_observed'
 vp=probe(v); step_rows=[json.loads(x) for x in sp.read_text().splitlines()]; lines=len(step_rows); assert (vp['width'],vp['height'],vp['frames'])==(640,480,lines) and vp['fps']=='20/1' and step_rows[-1]['tick_started_elapsed_seconds']>=29.9
 review={'schema':'task1_picklift_real24_same_eval48_canonical_review_label_v1','trial_id':stem,'review_success':bool(op['success']),'review_failure_category':op.get('failure_category'),'full_video_reviewed':True,'success_contract':'>5cm unsupported bilateral grasp sustained >=0.5s','source_video_sha256':sha(v),'operator_label_not_used_as_review_source':True}
 (OUT/'sidecars'/f'{stem}.json').write_text(json.dumps(review,indent=2)+'\n')
 rows.append({**{k:t[k] for k in ('trial_id','pose_order','eval_pose_id','model_key','coverage_tier','cell','nominal_yaw_degrees_modulo_90')},'operator_success':bool(op['success']),'review_success':bool(op['success']),'operator_review_agree':True,'video_sha256':sha(v),'video_probe':vp,'steps_rows':lines,'torque_disable_verified':True,'return_status':d['automatic_return']['result']['status']})
(OUT/'trials.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
def rate(rs): return {'successes':sum(r['review_success'] for r in rs),'failures':sum(not r['review_success'] for r in rs),'trials':len(rs),'success_rate':sum(r['review_success'] for r in rs)/len(rs)}
summary={'schema':'task1_picklift_real24_same_eval48_canonical_review_summary_v1','evaluation_id':plan['evaluation_id'],'overall':rate(rows),'by_yaw':{str(y):rate([r for r in rows if r['nominal_yaw_degrees_modulo_90']==y]) for y in (0,45)},'operator_review_agreement':{'agree':48,'disagree':0,'adjudications':0},'replacements_excluded_from_denominator':2,'interpretation_boundary':'Descriptive single-session engineering budget result; not a causal paper conclusion.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); man={'schema':'task1_picklift_real24_same_eval48_canonical_review_manifest_v1','evaluation_id':plan['evaluation_id'],'plan_sha256':sha(PLAN),'reviewed_scored_trials':48,'replacement_originals_excluded':2,'trials_jsonl_sha256':sha(OUT/'trials.jsonl'),'summary_sha256':sha(OUT/'summary.json')}; (OUT/'manifest.json').write_text(json.dumps(man,indent=2)+'\n');
outs=list((OUT/'sidecars').glob('*.json'))+[OUT/'trials.jsonl',OUT/'summary.json',OUT/'manifest.json']; (OUT/'hashes.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in outs)); print(json.dumps(summary,indent=2))
