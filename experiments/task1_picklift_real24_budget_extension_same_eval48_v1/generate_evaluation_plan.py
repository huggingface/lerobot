from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path

REPO=Path('/home/ubuntu24/Teleop/lerobot')
EXP=REPO/'experiments/task1_picklift_real24_budget_extension_same_eval48_v1'
OLD=REPO/'experiments/task1_picklift_real48_vs_real96_eval48_v1'
OLD_PLAN=OLD/'evaluation_plan.json'; OLD_SHA='7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1'
RESULT=EXP/'source_result_manifest.json'; RESULT_SHA='7940cf974e84eb11f90f8ea22ff1fe500a92c6dd1c6ad4255df39d0055512468'
SUBSET=Path('/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_budget_extension_v1/real24_subset_manifest.jsonl'); SUBSET_SHA='7cd5917bb03beafc347f9e1d6fd645e731eb0c26e6d1f0eaf98d7497e6d7d21f'
DATA=Path('/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real24_budget_extension_v1/accepted')
CKPT=Path('/home/ubuntu24/Teleop/artifacts/training/task1_picklift_real24_budget_extension_act_v1/full_100k/checkpoints/100000/pretrained_model')
MODEL_SHA='0b91103890fc78dc0fd3bff61457e4d06945041d68466f49f21bc6d40d4a9f29'; STATS_SHA='947d612d48280a98f3f6aeb37744e0aae9f8ea2034b6def24ebcf62c18ff4651'; DATA_SHA='c01c45f9dcaee557248bff997f3c244a9fdba2b6c13211821ee335d4bfee0712'

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x:raise RuntimeError(m)
def prompt(t):
 x=t['nominal_x_forward_m']*100;y=t['nominal_y_lateral_m']*100; yaw=t['nominal_yaw_degrees_modulo_90']; orient='方块边与任务网格线平行（0°）' if yaw==0 else '方块绕中心旋转45°（边与任务网格线成45°）'
 return f"摆放第{t['pose_order']}个冻结姿态：{t['cell']}（第{t['row']}行第{t['column']}列），红块中心放在任务网格 X={x:g} cm、Y={y:+g} cm 的交点；{orient}。"
def main():
 req(sha(OLD_PLAN)==OLD_SHA,'old plan');req(sha(RESULT)==RESULT_SHA,'research result');req(sha(SUBSET)==SUBSET_SHA,'subset')
 req(sha(CKPT/'model.safetensors')==MODEL_SHA,'model');req(sha(CKPT/'policy_preprocessor_step_3_normalizer_processor.safetensors')==STATS_SHA,'stats')
 audit=json.loads(Path('/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_budget_extension_act_v1/data_and_contract_v1/dataset_audit.json').read_text())['datasets']['real24'];req(audit['tree']['tree_sha256']==DATA_SHA and audit['episodes']==24 and audit['frames']==4263,'dataset')
 old=json.loads(OLD_PLAN.read_text()); subset_ids={json.loads(x)['plan_item_id'] for x in SUBSET.read_text().splitlines()}; trials=[]; counts={k:0 for k in ('seen_by_real24','added_by_real48','added_by_real96','unseen_by_both')}
 for i,src in enumerate(old['trials'][::2],1):
  if src.get('source_collection_plan_item_id') in subset_ids:tier='seen_by_real24'
  elif src['coverage_tier']=='seen_by_real48':tier='added_by_real48'
  else:tier=src['coverage_tier']
  counts[tier]+=1; t={**src,'order':i,'trial_id':f't{i:03d}_p{i:02d}_real24','artifact_stem':f't{i:03d}_p{i:02d}_real24','spawn_region':f't{i:03d}_p{i:02d}_real24','model_key':'real24','model_id':'ACT_Real24_budget_extension_seed1000_step100000','within_pair_order':1,'coverage_tier':tier};t['operator_placement_prompt_zh']=prompt(t);trials.append(t)
 req(counts=={'seen_by_real24':12,'added_by_real48':12,'added_by_real96':18,'unseen_by_both':6},f'membership {counts}')
 model={'model_id':'ACT_Real24_budget_extension_seed1000_step100000','checkpoint':str(CKPT),'model_sha256':MODEL_SHA,'processor_stats_sha256':STATS_SHA,'config_sha256':sha(CKPT/'config.json'),'train_config_sha256':sha(CKPT/'train_config.json'),'policy_preprocessor_sha256':sha(CKPT/'policy_preprocessor.json'),'dataset':str(DATA),'dataset_tree_sha256':DATA_SHA,'dataset_episodes':24,'dataset_frames':4263}
 plan={**old,'schema_version':1,'evaluation_id':'task1_picklift_real24_budget_extension_same_eval48_v1','status':'software_gate_frozen_hardware_not_authorized','purpose':'Evaluate the preselected Real24 budget checkpoint once on each unchanged frozen Eval48 pose.','comparison_role':'Task1 same-pose Real24 budget extension; old Real48/Real96 evidence remains external and immutable','research_contract':{'research_repo_commit':'06555f2','result_manifest':{'path':str(RESULT),'sha256':RESULT_SHA},'source_eval48_plan':{'path':str(OLD_PLAN),'sha256':OLD_SHA},'real24_subset_manifest':{'path':str(SUBSET),'sha256':SUBSET_SHA}},'evidence_root':'/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_budget_extension_same_eval48_v1','models':{'real24':model},'balance_invariants':{'poses':48,'trials':48,'coverage_tiers':counts,'cells_4_each':True,'yaw_0_45':{'0':24,'45':24},'pose_order_identical_to_source_eval48':True},'trials':trials,'authorization':{'hardware_authorized':False,'serial_accessed_during_preparation':False,'camera_accessed_during_preparation':False,'robot_accessed_during_preparation':False,'torque_accessed_during_preparation':False,'rollout_executed_during_preparation':False},'next_hardware_gate':'Stop before Follower 12 V; wait for explicit hardware GO.'}
 (EXP/'evaluation_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n'); print(json.dumps({'plan_sha256':sha(EXP/'evaluation_plan.json'),'membership':counts,'first_trial':trials[0]},indent=2))
if __name__=='__main__':main()
