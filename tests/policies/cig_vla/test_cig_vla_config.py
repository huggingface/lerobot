from lerobot.policies import CIGVLAConfig
from lerobot.policies.factory import get_policy_class, make_policy_config


def test_registration():
    assert isinstance(make_policy_config("cig_vla", device="cpu"), CIGVLAConfig)
    assert get_policy_class("cig_vla").name == "cig_vla"
