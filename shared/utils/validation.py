import logging

from lerobot.common.constants import HF_LEROBOT_CALIBRATION, default_calibration_path


def validate_calibration_directory():
    if HF_LEROBOT_CALIBRATION != default_calibration_path:
        logging.warning(
            f"Calibration directory is not the default ({default_calibration_path}): {HF_LEROBOT_CALIBRATION}"
        )
        logging.warning(
            f"Do not record datasets for team use with non-shared calibrations."
        )
