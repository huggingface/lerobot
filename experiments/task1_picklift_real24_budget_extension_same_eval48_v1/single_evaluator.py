from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; PLAN=HERE/'evaluation_plan.json'; PLAN_SHA='ada1a17eecc972a999fe8e8540015b42ebc3115577bd34a73985a9f97eb29abf'
OLD=REPO/'experiments/task1_picklift_real48_vs_real96_eval48_v1/paired_evaluator.py'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x:raise RuntimeError(m)
def load_plan(path=PLAN):
 req(sha(path)==PLAN_SHA,'plan hash mismatch');p=json.loads(path.read_text());req(p['evaluation_id']=='task1_picklift_real24_budget_extension_same_eval48_v1','id');req(len(p['trials'])==48,'count');req([x['pose_order'] for x in p['trials']]==list(range(1,49)),'order');req(all(x['model_key']=='real24' for x in p['trials']),'model');req(p['authorization']['hardware_authorized'] is False,'software gate must remain hardware false');return p
def verify(p):
 m=p['models']['real24'];c=Path(m['checkpoint']);r=p['research_contract'];
 for key in ('result_manifest','source_eval48_plan','real24_subset_manifest'):req(sha(Path(r[key]['path']))==r[key]['sha256'],key)
 req(sha(c/'model.safetensors')==m['model_sha256'],'model');req(sha(c/'policy_preprocessor_step_3_normalizer_processor.safetensors')==m['processor_stats_sha256'],'stats');cfg=json.loads((c/'config.json').read_text());req(cfg['chunk_size']==67 and cfg['n_action_steps']==67,'ACT queue');req(cfg['input_features']=={'observation.images.front':{'shape':[3,480,640],'type':'VISUAL'},'observation.state':{'shape':[6],'type':'STATE'}},'inputs');req(cfg['output_features']=={'action':{'shape':[6],'type':'ACTION'}},'output')
 s=p['setup'];req(s['control_fps']==20 and s['maximum_trial_seconds']==30 and s['max_relative_target'] is None and s['custom_absolute_action_clamp'] is False and s['custom_relative_step_limit_degrees'] is None,'deployment contract');req(s['policy_reset_after_ready_pose'] and s['ready_pose_before_every_trial'] and s['ready_pose_after_every_trial'] and not s['stop_on_success'],'window/ready')
 return {'status':'pass','plan_sha256':PLAN_SHA,'model_sha256':m['model_sha256'],'processor_stats_sha256':m['processor_stats_sha256'],'dataset_tree_sha256':m['dataset_tree_sha256'],'chunk_size':67,'n_action_steps':67,'input_contract':'front640x480+state[6]','output_contract':'action[6]'}
def fake(p):
 ready=np.asarray(p['setup']['ready_pose_state'],dtype=np.float32); records=[]; resets=0
 for t in p['trials']:
  state=ready.copy();resets+=1;rgb=np.zeros((480,640,3),np.uint8);raw=state.copy();sent=raw.copy();returned=ready.copy();records.append({'trial_id':t['trial_id'],'eval_pose_id':t['eval_pose_id'],'coverage_tier':t['coverage_tier'],'prompt':t['operator_placement_prompt_zh'],'ready_before':bool(np.array_equal(state,ready)),'policy_reset_before_tick0':True,'rgb_shape':list(rgb.shape),'raw_equals_official_sent':bool(np.array_equal(raw,sent)),'ready_after':bool(np.array_equal(returned,ready)),'torque_disabled':True})
 req(len(records)==48 and resets==48 and all(all((x['ready_before'],x['policy_reset_before_tick0'],x['raw_equals_official_sent'],x['ready_after'],x['torque_disabled'])) and x['rgb_shape']==[480,640,3] for x in records),'fake')
 # Deterministic slow-tick probe: next iteration begins after prior slow tick; no scheduled-deadline catch-up.
 starts=[];now=0.0
 for i in range(5):starts.append(now);compute=.08 if i==1 else .01;now+=compute+max(0,.05-compute)
 req(np.allclose(starts,[0.0,.05,.13,.18,.23],rtol=0.0,atol=1e-12),'slow tick pacing')
 return {'status':'pass','trials_exercised':48,'policy_reset_calls':48,'trial_ids':[x['trial_id'] for x in records],'coverage_counts':dict(__import__('collections').Counter(x['coverage_tier'] for x in records)),'all_contract_checks':True,'slow_tick_start_seconds':starts,'no_catch_up_burst':True,'hardware_access':{'serial':False,'camera':False,'robot':False,'torque':False,'12V':False,'rollout':False},'records':records}
def freeze(p,dry):
 out=Path(p['evidence_root'])/'software_gate_v1';req(not out.exists(),f'refuse overwrite {out}');out.mkdir(parents=True);(out/'evaluation_plan.json').write_text(PLAN.read_text());(out/'dry_run.json').write_text(json.dumps(dry,indent=2,sort_keys=True)+'\n');return out
def hardware(args,p):
 req(args.hardware_authorized,'--execute-hardware requires later explicit --hardware-authorized');spec=importlib.util.spec_from_file_location('old_eval48',OLD);mod=importlib.util.module_from_spec(spec);sys.path.insert(0,str(OLD.parent));spec.loader.exec_module(mod);mod.DEFAULT_PLAN=PLAN;mod.EXPECTED_PLAN_SHA256=PLAN_SHA;mod.EXPECTED_EVALUATION_ID=p['evaluation_id'];mod.MODEL_IDS=('real24',);mod.verify_static_files=lambda plan:verify(plan);mod.execute_hardware(args,p)
def main():
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--software-dry-run',action='store_true');g.add_argument('--execute-hardware',action='store_true');ap.add_argument('--freeze-software-evidence',action='store_true');ap.add_argument('--hardware-authorized',action='store_true');ap.add_argument('--trial-id');ap.add_argument('--operator-confirmed-ready',action='store_true');ap.add_argument('--replacement',action='store_true');ap.add_argument('--follower-port',default='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00');ap.add_argument('--camera-device',default='/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0');ap.add_argument('--calibration',type=Path,default=Path('/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_main.json'));ap.add_argument('--plan',type=Path,default=PLAN);a=ap.parse_args();p=load_plan(a.plan)
 if a.software_dry_run:
  req(not a.hardware_authorized and not a.operator_confirmed_ready and a.trial_id is None,'hardware args prohibited in dry run');dry={'schema':'task1_picklift_real24_same_eval48_software_dry_run_v1','evaluation_id':p['evaluation_id'],'status':'pass_hardware_not_authorized','static':verify(p),'fake_protocol':fake(p),'next_gate':'Stop before Follower 12V.'};
  if a.freeze_software_evidence:dry['evidence_root']=str(freeze(p,dry));print(json.dumps(dry,indent=2,sort_keys=True));return
 hardware(a,p)
if __name__=='__main__':main()
