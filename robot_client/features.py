from typing import Any

OBS_STR = "observation"


def map_robot_keys_to_lerobot_features(robot: Any) -> dict[str, dict]:
    """
    Преобразует спецификацию робота в словарь признаков формата LeRobot,
    совместимый с сервером (используется на стороне сервера для сборки кадров).
    """
    features: dict[str, dict] = {}

    joint_names = [k for k, v in robot.observation_features.items() if v is float]
    if joint_names:
        features[f"{OBS_STR}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        }

    for key, value in robot.observation_features.items():
        if isinstance(value, tuple) and len(value) == 3:
            h, w, c = value
            features[f"{OBS_STR}.images.{key}"] = {
                "dtype": "image",
                "shape": (h, w, c),
                "names": ["height", "width", "channels"],
            }

    return features
