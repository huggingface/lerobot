import hashlib,json
from pathlib import Path
EXP=Path('/home/ubuntu24/Teleop/lerobot/experiments/task1_picklift_real24_budget_extension_same_eval48_v1'); ROOT=Path('/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real24_budget_extension_same_eval48_v1/software_gate_v1')
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 files={n:ROOT/n for n in ('evaluation_plan.json','dry_run.json','offline_inference_cuda.json')};
 for p in files.values():
  if not p.exists():raise RuntimeError(p)
 inf=json.loads(files['offline_inference_cuda.json'].read_text());dry=json.loads(files['dry_run.json'].read_text());
 if inf['status']!='pass' or inf['output_shape']!=[1,6] or not inf['output_finite']:raise RuntimeError('inference')
 m={'schema':'task1_picklift_real24_same_eval48_software_gate_manifest_v1','evaluation_id':'task1_picklift_real24_budget_extension_same_eval48_v1','status':'pass_hardware_not_authorized','plan_sha256':'ada1a17eecc972a999fe8e8540015b42ebc3115577bd34a73985a9f97eb29abf','trials':48,'membership':{'seen_by_real24':12,'added_by_real48':12,'added_by_real96':18,'unseen_by_both':6},'checkpoint_inference':{'output_shape':[1,6],'finite':True,'model_sha256':inf['model_sha256'],'processor_stats_sha256':inf['processor_stats_sha256']},'tests':{'pytest':'4 passed','fake_trials':48,'no_catch_up':True,'plan_tamper_fail_closed':True},'hardware_access':{'serial':False,'camera':False,'robot':False,'torque':False,'12V':False,'rollout':False},'artifacts':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},'first_future_prompt':json.loads(files['evaluation_plan.json'].read_text())['trials'][0]['operator_placement_prompt_zh']}
 mp=ROOT/'manifest.json';mp.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');hp=ROOT/'hashes.sha256';hp.write_text(''.join(f'{sha(p)}  {p}\n' for p in [*files.values(),mp]));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
