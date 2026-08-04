from __future__ import annotations

import hashlib, json, re, subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

EXP=Path('/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_real48_vs_real96_eval48_v1')
PLAN=EXP/'evaluation_plan.json'
ROOT=Path('/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real48_vs_real96_eval48_v1/trials')
OUT=ROOT.parent/'canonical_video_review_v1'
PLAN_SHA='7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1'
MISSED={'t001_p01_real48','t006_p03_real96','t017_p09_real48','t041_p21_real48','t048_p24_real48','t049_p25_real48','t059_p30_real96','t063_p32_real96','t064_p32_real48','t083_p42_real96','t084_p42_real48','t096_p48_real48'}
SPATIAL={'t026_p13_real96','t029_p15_real48','t034_p17_real96','t046_p23_real96','t047_p24_real96','t050_p25_real96','t051_p26_real96','t052_p26_real48','t062_p31_real96','t067_p34_real96','t068_p34_real48','t071_p36_real96','t072_p36_real48','t075_p38_real96','t086_p43_real96'}
FAIL=MISSED|SPATIAL

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x: raise RuntimeError(m)
def probe(p):
 q=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(p)],check=True,capture_output=True,text=True)
 s=json.loads(q.stdout)['streams'][0]
 return {'width':int(s['width']),'height':int(s['height']),'avg_frame_rate':s['avg_frame_rate'],'frames':int(s['nb_frames']),'duration_seconds':float(s['duration'])}
def steps(p):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 req(rows and [r['step'] for r in rows]==list(range(len(rows))),f'non-contiguous steps {p}')
 return {'lines':len(rows),'first_tick_elapsed_seconds':rows[0]['tick_started_elapsed_seconds'],'last_tick_elapsed_seconds':rows[-1]['tick_started_elapsed_seconds']}
def rate(rows):
 n=len(rows); s=sum(r['review_success'] for r in rows)
 return {'successes':s,'failures':n-s,'trials':n,'success_rate':s/n}

def main():
 req(not OUT.exists(),f'refusing overwrite {OUT}')
 req(sha(PLAN)==PLAN_SHA,'plan hash mismatch')
 plan=json.loads(PLAN.read_text()); trials=plan['trials']; req(len(trials)==96,'plan != 96')
 req([t['trial_id'] for t in trials]==[f't{i:03d}_p{(i+1)//2:02d}_{t["model_key"]}' for i,t in enumerate(trials,1)],'trial/order identity mismatch')
 req(len({t['trial_id'] for t in trials})==96,'duplicate trial id')
 OUT.mkdir(); (OUT/'sidecars').mkdir(); rows=[]; input_hashes={str(PLAN):PLAN_SHA}
 for t in trials:
  stem=t['artifact_stem']; mainp=ROOT/f'{stem}.json'; opp=ROOT/f'{stem}.operator_label.json'; video=ROOT/f'{stem}.mp4'; sp=ROOT/f'{stem}.steps.jsonl'; ready=ROOT/f'{stem}.ready.jsonl'; ret=ROOT/f'{stem}.return.jsonl'; pair=ROOT/f'{stem}.paired_eval48.json'
  for p in (mainp,opp,video,sp,ready,ret,pair): req(p.exists(),f'missing {p}')
  d=json.loads(mainp.read_text()); op=json.loads(opp.read_text()); pd=json.loads(pair.read_text())
  req(d['status']=='completed_pending_operator_annotation' and d['run_error'] is None and d['torque_disable_verified'] is True,f'invalid trial {stem}')
  req(pd['canonical_video_review_label']['status']=='pending','pre-existing review mutation')
  req(d['paired_eval48_plan_sha256']==PLAN_SHA and d['paired_eval48_trial']['trial_id']==stem,'trial plan identity mismatch')
  req(d['upstream_action_modified_events']==0,'upstream action modified')
  for p,key,obj in ((video,'video',d),(sp,'steps_jsonl',d)):
   req(sha(p)==obj[key]['sha256'],f'hash mismatch {p}')
  vp=probe(video); st=steps(sp)
  req((vp['width'],vp['height'])==(640,480) and Fraction(vp['avg_frame_rate'])==20,f'video contract {stem}')
  req(vp['frames']==st['lines']==d['video']['frames']==d['steps_jsonl']['lines'],f'frame/steps mismatch {stem}')
  req(st['last_tick_elapsed_seconds']>=29.9,f'short wall window {stem}')
  req(d['ready_pose_alignment']['result']['status']=='ready_pose_observed' and d['automatic_return']['result']['status']=='ready_pose_observed',f'ready/return {stem}')
  req(sum(1 for _ in ready.open())==d['ready_pose_alignment']['trajectory']['lines'],f'ready lines {stem}')
  req(sum(1 for _ in ret.open())==d['automatic_return']['trajectory']['lines'],f'return lines {stem}')
  success=stem not in FAIL; category=None if success else ('missed_grasp' if stem in MISSED else 'spatial_offset')
  visible=('Canonical video shows bilateral grasp followed by an unsupported lift strictly above 5 cm sustained for at least 0.5 s.' if success else ('Canonical video shows grasp attempts without a qualifying unsupported lift; no successful bilateral grasp is visible.' if category=='missed_grasp' else 'Canonical video shows grasp closure or approach spatially offset from the cube, with no qualifying unsupported lift.'))
  review={'schema':'task1_picklift_eval48_canonical_video_review_label_v1','review_id':'task1_picklift_real48_vs_real96_eval48_canonical_video_review_v1','evaluation_id':plan['evaluation_id'],'trial_id':stem,'status':'reviewed','review_success':success,'review_failure_category':category,'success_contract':'>5cm unsupported bilateral grasp sustained >=0.5s within full policy window','full_video_reviewed':True,'review_visible_evidence':visible,'source_video':str(video),'source_video_sha256':sha(video),'operator_label_not_used_as_review_source':True}
  side=OUT/'sidecars'/f'{stem}.canonical_review.json'; side.write_text(json.dumps(review,indent=2,sort_keys=True)+'\n')
  agree=bool(op['success'])==success
  row={**{k:t[k] for k in ('trial_id','pose_order','eval_pose_id','model_key','model_id','coverage_tier','cell','nominal_yaw_degrees_modulo_90','within_pair_order')},'operator_success':bool(op['success']),'operator_failure_category':op.get('failure_category'),'review_success':success,'review_failure_category':category,'operator_review_agree':agree,'adjudication_path':None,'video_path':str(video),'video_sha256':sha(video),'steps_path':str(sp),'steps_sha256':sha(sp),'video_probe':vp,'steps_probe':st,'ready_status':d['ready_pose_alignment']['result']['status'],'return_status':d['automatic_return']['result']['status'],'torque_disable_verified':True,'upstream_action_modified_events':0,'review_sidecar':str(side),'review_sidecar_sha256':sha(side)}
  rows.append(row)
  for p in (mainp,opp,video,sp,ready,ret,pair): input_hashes[str(p)]=sha(p)
 req(all(r['operator_review_agree'] for r in rows),'disagreement requires adjudication before freeze')
 trials_jsonl=OUT/'trials.jsonl'; trials_jsonl.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 models={}
 for m in ('real48','real96'):
  rs=[r for r in rows if r['model_key']==m]; models[m]=rate(rs); models[m]['failure_categories']=dict(Counter(r['review_failure_category'] for r in rs if not r['review_success']))
 tiers={}; cells={}; yaws={}
 for tier in ('seen_by_real48','added_by_real96','unseen_by_both'):
  tiers[tier]={m:rate([r for r in rows if r['model_key']==m and r['coverage_tier']==tier]) for m in ('real48','real96')}
 for cell in sorted({r['cell'] for r in rows}): cells[cell]={m:rate([r for r in rows if r['model_key']==m and r['cell']==cell]) for m in ('real48','real96')}
 for yaw in (0,45): yaws[str(yaw)]={m:rate([r for r in rows if r['model_key']==m and r['nominal_yaw_degrees_modulo_90']==yaw]) for m in ('real48','real96')}
 pairs=[]; pc=Counter()
 for po in range(1,49):
  pair={r['model_key']:r for r in rows if r['pose_order']==po}; req(set(pair)=={'real48','real96'},f'bad pair {po}')
  a=pair['real48']['review_success']; b=pair['real96']['review_success']; outcome='both_success' if a and b else 'both_failure' if not a and not b else 'real48_only' if a else 'real96_only'; pc[outcome]+=1
  pairs.append({'pose_order':po,'eval_pose_id':pair['real48']['eval_pose_id'],'coverage_tier':pair['real48']['coverage_tier'],'cell':pair['real48']['cell'],'yaw':pair['real48']['nominal_yaw_degrees_modulo_90'],'real48_success':a,'real96_success':b,'real96_minus_real48':int(b)-int(a),'outcome':outcome})
 summary={'schema':'task1_picklift_real48_vs_real96_eval48_canonical_review_summary_v1','review_id':'task1_picklift_real48_vs_real96_eval48_canonical_video_review_v1','evaluation_id':plan['evaluation_id'],'status':'review_complete','overall':models,'by_coverage_tier':tiers,'by_cell':cells,'by_yaw':yaws,'paired_outcomes':dict(pc),'paired_results':pairs,'paired_real96_minus_real48':{'success_count_difference':models['real96']['successes']-models['real48']['successes'],'success_rate_difference':models['real96']['success_rate']-models['real48']['success_rate']},'operator_review_agreement':{'agree':sum(r['operator_review_agree'] for r in rows),'disagree':sum(not r['operator_review_agree'] for r in rows),'adjudications':0},'evidence_contract':{'trials':96,'videos':96,'all_640x480_at_20fps':True,'all_video_frames_equal_steps_rows':True,'all_policy_windows_reach_29p9_seconds':True,'all_ready_return_torque_and_action_path_valid':True,'duplicates':0,'missing':0,'replacements':0},'interpretation_boundary':'Descriptive single-session engineering result; not a causal paper conclusion.'}
 sump=OUT/'summary.json'; sump.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 manifest={'schema':'task1_picklift_real48_vs_real96_eval48_canonical_review_manifest_v1','review_id':summary['review_id'],'evaluation_id':plan['evaluation_id'],'plan_path':str(PLAN),'plan_sha256':PLAN_SHA,'reviewed_trials':96,'review_sidecars':96,'operator_review_disagreements':0,'adjudications':0,'trials_jsonl':str(trials_jsonl),'trials_jsonl_sha256':sha(trials_jsonl),'summary':str(sump),'summary_sha256':sha(sump),'immutable_input_hash_count':len(input_hashes),'immutable_input_hashes':input_hashes}
 manp=OUT/'manifest.json'; manp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 outputs=[*sorted((OUT/'sidecars').glob('*.json')),trials_jsonl,sump,manp]
 hp=OUT/'hashes.sha256'; hp.write_text(''.join(f'{sha(p)}  {p}\n' for p in outputs))
 print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
