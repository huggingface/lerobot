import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_budget_extension_same_eval48_v1/trials'); SRC=ROOT.parent/'blind_review_sources_v2'; OUT=ROOT.parent/'canonical_video_review_v2'; PLAN=Path('/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_real24_budget_extension_same_eval48_v1/evaluation_plan.json')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def probe(p):
 s=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(p)],capture_output=True,text=True,check=True).stdout)['streams'][0]; return {'width':int(s['width']),'height':int(s['height']),'fps':s['avg_frame_rate'],'frames':int(s['nb_frames']),'duration':float(s['duration'])}
assert not OUT.exists() and sha(PLAN)=='ada1a17eecc972a999fe8e8540015b42ebc3115577bd34a73985a9f97eb29abf'; plan=json.loads(PLAN.read_text()); assert len(plan['trials'])==48
files={'review-001-016.json':'4474ed6ffffd84e0cf47896632c9cfc7a652045b460415525e45dcfb7cbcda49','review-017-032.json':'66aa97387bf76154c9ddc7922f56e475855a7ec92b198a5b446775f9b76d141d','review-033-048.json':'11e00557b75a730b984847124536a558c83d69e29dd26aa0ac9733e0101d126b'}
blind=[]
for n,h in files.items(): assert sha(SRC/n)==h; blind+=json.loads((SRC/n).read_text())
assert len(blind)==48 and len({r['trial_id'] for r in blind})==48
OUT.mkdir(); (OUT/'sidecars').mkdir(); rows=[]
for t,b in zip(plan['trials'],sorted(blind,key=lambda x:int(x['trial_id'][1:4]))):
 stem=t['artifact_stem']; assert b['trial_id']==stem; scored=stem+'__replacement1' if stem in ('t017_p17_real24','t022_p22_real24') else stem; d=json.loads((ROOT/f'{scored}.json').read_text()); v=ROOT/f'{scored}.mp4'; sp=ROOT/f'{scored}.steps.jsonl'; vp=probe(v); sr=[json.loads(x) for x in sp.read_text().splitlines()]; assert vp['frames']==len(sr) and vp['width']==640 and vp['height']==480 and vp['fps']=='20/1' and sr[-1]['tick_started_elapsed_seconds']>=29.9
 side={'schema':'task1_picklift_real24_same_eval48_canonical_review_v2_sidecar_v1','trial_id':stem,'scored_stem':scored,'review_success':b['success'],'failure_category':b.get('failure_category'),'evidence_intervals':b['evidence_intervals'],'visible_grasp_lift_hold_observations':b['visible_grasp_lift_hold_observations'],'confidence':b['confidence'],'source_video_sha256':sha(v),'blind_source_frozen_before_operator_join':True}
 (OUT/'sidecars'/f'{stem}.json').write_text(json.dumps(side,indent=2)+'\n'); rows.append({**{k:t[k] for k in ('trial_id','pose_order','eval_pose_id','model_key','coverage_tier','cell','nominal_yaw_degrees_modulo_90')},'blind_success':b['success'],'blind_failure_category':b.get('failure_category'),'blind_confidence':b['confidence'],'evidence_intervals':b['evidence_intervals'],'video_sha256':sha(v),'video_probe':vp,'steps_rows':len(sr),'ready_status':d['ready_pose_alignment']['result']['status'],'return_status':d['automatic_return']['result']['status'],'torque_disable_verified':d['torque_disable_verified']})
(OUT/'blind_trials.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)); blind_hash=sha(OUT/'blind_trials.jsonl')
op={};
for p in ROOT.glob('*.operator_label.json'):
 x=json.loads(p.read_text()); op[x['trial_id']]=x
for r in rows:
 stem=r['trial_id']; scored=stem+'__replacement1' if stem in ('t017_p17_real24','t022_p22_real24') else stem; o=op[scored]; r['operator_success']=o['success']; r['operator_failure_category']=o.get('failure_category'); r['operator_review_agree']=r['blind_success']==o['success']
assert all(r['operator_review_agree'] for r in rows)
(OUT/'trials.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
def rate(rs): return {'successes':sum(r['blind_success'] for r in rs),'failures':sum(not r['blind_success'] for r in rs),'trials':len(rs),'success_rate':sum(r['blind_success'] for r in rs)/len(rs)}
summary={'schema':'task1_picklift_real24_same_eval48_canonical_review_v2_summary_v1','evaluation_id':plan['evaluation_id'],'overall':rate(rows),'by_yaw':{str(y):rate([r for r in rows if r['nominal_yaw_degrees_modulo_90']==y]) for y in (0,45)},'operator_review_agreement':{'agree':48,'disagree':0,'adjudications':0},'blind_source_hashes':files,'replacement_originals_excluded':2,'v1_invalid_unused':True,'interpretation_boundary':'Descriptive single-session engineering result; not a causal paper conclusion.'}; (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); man={'schema':'task1_picklift_real24_same_eval48_canonical_review_v2_manifest_v1','evaluation_id':plan['evaluation_id'],'plan_sha256':sha(PLAN),'blind_files_frozen_before_operator_join':True,'blind_source_hashes':files,'reviewed_scored_trials':48,'replacement_originals_excluded':2,'blind_trials_sha256':blind_hash,'trials_sha256':sha(OUT/'trials.jsonl'),'summary_sha256':sha(OUT/'summary.json'),'v1_invalid_unused':True}; (OUT/'manifest.json').write_text(json.dumps(man,indent=2)+'\n'); outs=list((OUT/'sidecars').glob('*.json'))+[OUT/'blind_trials.jsonl',OUT/'trials.jsonl',OUT/'summary.json',OUT/'manifest.json']; (OUT/'hashes.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in outs)); print(json.dumps(summary,indent=2))
