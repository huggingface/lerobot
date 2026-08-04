import importlib.util
from pathlib import Path
P=Path(__file__).with_name('single_evaluator.py');s=importlib.util.spec_from_file_location('r',P);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def test_plan_and_membership():
 p=r.load_plan();assert [x['coverage_tier'] for x in p['trials']].count('seen_by_real24')==12;assert [x['coverage_tier'] for x in p['trials']].count('added_by_real48')==12;assert [x['coverage_tier'] for x in p['trials']].count('added_by_real96')==18;assert [x['coverage_tier'] for x in p['trials']].count('unseen_by_both')==6
def test_fake_48_and_no_catchup():
 x=r.fake(r.load_plan());assert x['trials_exercised']==48 and x['no_catch_up_burst'] and x['all_contract_checks']
def test_static_hashes():assert r.verify(r.load_plan())['status']=='pass'
def test_plan_tamper_fails_closed(tmp_path):
 p=tmp_path/'plan.json';p.write_bytes(r.PLAN.read_bytes()+b' ')
 try:r.load_plan(p)
 except RuntimeError:return
 assert False,'tampered plan was accepted'
