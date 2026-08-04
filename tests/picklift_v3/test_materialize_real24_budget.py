from examples.picklift_v3.materialize_real24_budget import PLAN_ITEM_IDS, validate_balance
from examples.picklift_v3.real96_plan import real96_items


def test_frozen_real24_ids_are_unique_real48_members_and_balanced():
    by_id = {item["plan_item_id"]: item for item in real96_items()}
    selection = [by_id[item_id] for item_id in PLAN_ITEM_IDS]
    assert len(PLAN_ITEM_IDS) == len(set(PLAN_ITEM_IDS)) == 24
    assert all(item["real48_member"] and item["subset_role"] == "core" for item in selection)
    checks = validate_balance(selection)
    assert set(checks["cells"].values()) == {2}
    assert set(checks["sessions"].values()) == {6}
