from lerobot.utils.import_utils import register_third_party_devices


def main() -> None:
    register_third_party_devices()
    from lerobot.scripts.lerobot_record import main as record_main

    record_main()


