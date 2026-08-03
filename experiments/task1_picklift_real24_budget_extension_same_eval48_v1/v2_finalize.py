import json,glob,hashlib
from pathlib import Path
R=Path('/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_budget_extension_same_eval48_v1'); S=R/'blind_review_sources_v2'; T=R/'trials'; O=R/'canonical_video_review_v2'; P=Path('/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_real24_budget_extension_same_eval48_v1/evaluation_plan.json')
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
names=['review-001-016.json','review-017-032.json','review-033-048.json']; hashes=['4474ed6ffffd84e0cf47896632c9cfc7a652045b460415525e45dcfb7cbcda49','66aa97387bf76154c9ddc7922f56e475855a7ec92b198a5b446775f9b76d141d','11e00557b75a730b984847124536a558c83d69e29dd26aa0ac9733e0101d126b']; blind=[]
for n,x in zip(names,hashes): assert h(S/n)==x; blind+=json.loads((S/n).read_text())
blind.sort(key=lambda x:int(x['trial_id'][1:4])); assert len(blind)==48; O.mkdir(); (O/'sidecars').mkdir()
rows=[]
for t,b in zip(json.loads(P.read_text())['trials'],blind):
 stem=t['artifact_stem']; scored=stem+'__replacement1' if stem in ('t017_p17_real24','t022_p22_real24') else stem
 d=json.loads((T/f'{scored}.json').read_text()); video=T/f'{scored}.mp4'; steps=T/f'{scored}.steps.jsonl'; assert d['torque_disable_verified'] and d['automatic_return']['result']['status']=='ready_pose_observed'
 if 'observations' in b: ev=b['evidence_intervals_seconds']; obs=b['observations']
 elif 'visible_observations' in b: ev=b['evidence_intervals_seconds']; obs=b['visible_observations']
 else: ev=b['evidence_intervals']; obs=b['visible_grasp_lift_hold_observations']
 side={'trial_id':stem,'base_trial_id':stem,'scored_stem':scored,'blind_success':b['success'],'blind_failure_category':b.get('failure_category'),'evidence_intervals':ev,'visible_observations':obs,'confidence':b['confidence'],'raw_blind_record':b,'video_sha256':h(video)}; (O/'sidecars'/f'{stem}.json').write_text(json.dumps(side,indent=2)+'\n'); rows.append({**{k:t[k] for k in ('trial_id','pose_order','eval_pose_id','model_key','coverage_tier','cell','nominal_yaw_degrees_modulo_90')},'base_trial_id':stem,'scored_stem':scored,'blind_success':b['success'],'blind_failure_category':b.get('failure_category'),'evidence_intervals':ev,'video_sha256':h(video),'steps_sha256':h(steps)})
(O/'blind_trials.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows)); blind_hash=h(O/'blind_trials.jsonl')
op={json.loads(p.read_text())['trial_id']:json.loads(p.read_text()) for p in T.glob('*.operator_label.json')}
for r in rows: r['operator_success']=op[r['scored_stem']]['success']; r['operator_failure_category']=op[r['scored_stem']].get('failure_category'); r['operator_review_agree']=r['blind_success']==r['operator_success']
(O/'trials.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows)); dis=[r['trial_id'] for r in rows if not r['operator_review_agree']]
def rate(a): return {'successes':sum(x['blind_success'] for x in a),'failures':sum(not x['blind_success'] for x in a),'trials':len(a),'success_rate':sum(x['blind_success'] for x in a)/len(a)}
summary={'overall':rate(rows),'by_yaw':{str(y):rate([x for x in rows if x['nominal_yaw_degrees_modulo_90']==y]) for y in (0,45)},'operator_review_agreement':{'agree':48-len(dis),'disagree':len(dis),'adjudications':0},'blind_source_hashes':dict(zip(names,hashes)),'cross_check_backup_sha256':'e493d7fcbf135d0f5a08add8c85da9f9f23195e573b4102a1a0414a6c5cf893f','v1_invalid_unused':True}; (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); manifest={'blind_frozen_before_operator_join':True,'blind_trials_sha256':blind_hash,'trials_sha256':h(O/'trials.jsonl'),'summary_sha256':h(O/'summary.json'),'disagreements':dis,'replacement_originals_excluded':2}; (O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); fs=list((O/'sidecars').glob('*.json'))+[O/'blind_trials.jsonl',O/'trials.jsonl',O/'summary.json',O/'manifest.json']; (O/'hashes.sha256').write_text(''.join(f'{h(p)}  {p}\n' for p in fs)); print(summary)
