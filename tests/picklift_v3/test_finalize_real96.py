from examples.picklift_v3.finalize_real96 import validate_global
from examples.picklift_v3.real96_plan import real96_items


def test_validate_global_accepts_frozen_real96_and_two_retained_discards():
    accepted = []
    ledger = []
    for item in real96_items():
        row = {
            **item,
            "attempt_id": f"{item['plan_item_id']}_attempt_01",
            "accepted_episode_id": f"{item['session_id']}:{item['session_order']}",
            "accepted": True,
            "result": "success",
        }
        accepted.append(row)
        ledger.append(row)
    for suffix, item in enumerate((accepted[2], accepted[36]), 1):
        ledger.append(
            {
                **item,
                "attempt_id": f"{item['plan_item_id']}_discard_{suffix}",
                "accepted_episode_id": None,
                "accepted": False,
                "result": "discard",
            }
        )
    balances = validate_global(ledger, accepted)
    assert set(balances["real96"]["cells"].values()) == {8}
    assert set(balances["real48"]["cells"].values()) == {4}
