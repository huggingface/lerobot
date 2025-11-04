from lerobot.utils.import_utils import register_third_party_devices


def main() -> None:
    # Ensure third-party devices (like Piper) are registered before CLI parser builds choices
    register_third_party_devices()
    from lerobot.scripts.lerobot_teleoperate import main as tele_main

    tele_main()


