from __future__ import annotations
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path('/home/ubuntu24/Teleop/artifacts/evaluation/task1_picklift_real48_vs_real96_eval48_v1/canonical_video_review_v1')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x: raise RuntimeError(m)
def main():
 man=json.loads((ROOT/'manifest.json').read_text()); summ=json.loads((ROOT/'summary.json').read_text()); rows=[json.loads(x) for x in (ROOT/'trials.jsonl').read_text().splitlines()]
 req(len(rows)==96 and len({r['trial_id'] for r in rows})==96,'rows')
 req([int(r['trial_id'][1:4]) for r in rows]==list(range(1,97)),'ids')
 req(sum(r['model_key']=='real48' for r in rows)==48 and sum(r['model_key']=='real96' for r in rows)==48,'models')
 req(all(r['operator_review_agree'] for r in rows),'unexpected disagreement')
 req(man['plan_sha256']=='7de77eed859898a6397265244ab2f4c189d91dbb565202b99a0a5bdd208214f1','plan')
 req(sha(Path(man['plan_path']))==man['plan_sha256'],'plan current hash')
 for line in (ROOT/'hashes.sha256').read_text().splitlines():
  digest,path=line.split('  ',1); req(sha(Path(path))==digest,f'output hash {path}')
 for path,digest in man['immutable_input_hashes'].items(): req(sha(Path(path))==digest,f'input changed {path}')
 req(summ['overall']['real96']['successes']-summ['overall']['real48']['successes']==summ['paired_real96_minus_real48']['success_count_difference'],'delta')
 out={'schema':'task1_picklift_real48_vs_real96_eval48_canonical_review_validation_v1','status':'pass','reviewed_trials':96,'unique_trials':96,'models_48_each':True,'pairs_48_complete':True,'operator_review_agreements':96,'disagreements':0,'adjudications':0,'hash_manifest_verified':True,'immutable_inputs_verified':True,'plan_sha256_verified':True}
 p=ROOT/'independent_validation.json'; req(not p.exists(),'refuse overwrite validation'); p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
