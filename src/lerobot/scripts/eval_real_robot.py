#!/usr/bin/env python
"""
실제 로봇에서 정책을 평가하기 위한 간단한 스크립트.
lerobot-record보다 단순하며, 에피소드 저장 여부를 키보드로 직접 제어할 수 있습니다.

키보드 조작:
  → (오른쪽 화살표) : 현재 에피소드 조기 종료 후 저장 여부 선택
  s                  : 에피소드 성공으로 저장
  d                  : 에피소드 실패/폐기 (저장 안 함)
  ESC                : 전체 평가 종료

사용 예시:
    python -m lerobot.scripts.eval_real_robot \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM1 \\
        --robot.id=my_follower \\
        --robot.cameras="{'top': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG', 'warmup_s': 5}, 'hand': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG', 'warmup_s': 5}}" \\
        --policy.path=outputs/train/act_marker_to_cup_v2_merged_diverse/checkpoints/last/pretrained_model \\
        --task="Pick up the marker and place it into the cup" \\
        --num_episodes=10 \\
        --episode_time_s=15 \\
        --reset_time_s=20 \\
        --fps=30 \\
        --save_dataset=false
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say


@dataclass
class EvalRealRobotConfig:
    robot: RobotConfig
    # 결과 저장 repo 이름 (--repo_name으로 전달)
    repo_name: str = "eval_real_robot"
    # 정책 경로 (--policy.path로 전달)
    policy: PreTrainedConfig | None = None
    # 수행할 태스크 설명
    task: str = "Perform the task"
    # 평가할 에피소드 수
    num_episodes: int = 60
    # 에피소드당 실행 시간 (초)
    episode_time_s: float = 30.0
    # 리셋 대기 시간 (초)
    reset_time_s: float = 20.0
    # 제어 FPS
    fps: int = 30
    # 데이터셋 저장 여부 (False면 에피소드 결과만 출력)
    save_dataset: bool = True
    # 데이터셋 저장 경로 (save_dataset=True일 때만 사용)
    dataset_repo_id: str = "data/eval_results"
    # policy 메타 로딩에 사용할 기존 학습 데이터셋 repo (save_dataset=False일 때 임시 폴더 대신 사용)
    train_dataset_repo_id: str = ""
    # 사운드 재생 여부
    play_sounds: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("--policy.path 를 반드시 지정해야 합니다.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def run_eval_episode(
    robot: Robot,
    policy,
    preprocessor,
    postprocessor,
    dataset_features: dict,
    events: dict,
    fps: int,
    episode_time_s: float,
    task: str,
    dataset: LeRobotDataset | None = None,
) -> dict:
    """
    단일 에피소드를 실행합니다.

    Returns:
        dict: {
            "num_steps": int,          # 실행된 스텝 수
            "exit_early": bool,        # 조기 종료 여부
            "error": str | None,       # 발생한 에러 메시지 (없으면 None)
        }
    """
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    num_steps = 0
    timestamp = 0.0
    start_t = time.perf_counter()

    result = {"num_steps": 0, "exit_early": False, "error": None}

    while timestamp < episode_time_s:
        loop_start_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            result["exit_early"] = True
            logging.info("조기 종료 요청됨 (→ 키).")
            break

        try:
            obs = robot.get_observation()
        except Exception as e:
            result["error"] = f"get_observation 실패: {e}"
            logging.error(result["error"])
            break

        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        try:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
        except Exception as e:
            result["error"] = f"predict_action 실패: {e}"
            logging.error(result["error"])
            break

        act_processed: dict = make_robot_action(action_values, dataset_features)
        robot_action_to_send = robot_action_processor((act_processed, obs))

        # 에피소드 첫 스텝의 정책 명령을 출력 → HOME 후보값으로 활용
        if num_steps == 0:
            first_pose = {k: round(float(v), 3) for k, v in robot_action_to_send.items() if k.endswith(".pos")}
            logging.info(f"[FIRST_ACTION] {first_pose}")

        try:
            robot.send_action(robot_action_to_send)
        except Exception as e:
            result["error"] = f"send_action 실패: {e}"
            logging.error(result["error"])
            break

        if dataset is not None:
            action_frame = build_dataset_frame(dataset_features, act_processed, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": task}
            dataset.add_frame(frame)

        num_steps += 1
        dt_s = time.perf_counter() - loop_start_t
        sleep_time_s = 1.0 / fps - dt_s
        # 첫 스텝은 GPU 워밍업으로 항상 느리므로 무시, 이후에도 5% 여유 허용
        if sleep_time_s < 0 and num_steps > 1 and dt_s > (1.0 / fps) * 1.05:
            logging.warning(
                f"[step {num_steps}] 루프 느림: {1/dt_s:.1f} Hz (목표 {fps} Hz). "
                "카메라 FPS 또는 policy inference 시간 확인 필요."
            )
        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_t

    result["num_steps"] = num_steps
    return result


def wait_for_reset(
    robot: Robot,
    events: dict,
    fps: int,
    reset_time_s: float,
    task: str,
) -> bool:
    """
    리셋 대기 구간. 로봇은 정지 상태. 사용자가 수동으로 리셋합니다.
    ESC를 누르면 False를 반환 (전체 종료).
    """
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    logging.info(f"환경 리셋 중... ({reset_time_s:.0f}초 대기) 로봇을 수동으로 초기 위치로 돌려놓으세요.")
    log_say("Reset the environment", True)

    start_t = time.perf_counter()
    timestamp = 0.0

    while timestamp < reset_time_s:
        loop_start_t = time.perf_counter()

        if events["stop_recording"]:
            return False
        if events["exit_early"]:
            events["exit_early"] = False
            logging.info("리셋 구간 조기 종료.")
            break

        # 리셋 구간에서는 로봇 관찰만 수집 (액션 없음)
        try:
            robot.get_observation()
        except Exception:
            pass

        dt_s = time.perf_counter() - loop_start_t
        precise_sleep(max(1.0 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_t

    return True


@parser.wrap()
def eval_real_robot(cfg: EvalRealRobotConfig):
    init_logging()
    logging.info(pformat(vars(cfg)))

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot = make_robot_from_config(cfg.robot)

    # 데이터셋 feature 구성 (policy 로딩에 meta 필요)
    # 주의: use_videos=True 여야 카메라 feature가 포함됨 (False이면 이미지 feature 전체 제외됨)
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    # dataset 생성 또는 기존 학습 데이터셋 로드
    if cfg.save_dataset:
        dataset = LeRobotDataset.create(
            cfg.dataset_repo_id,
            cfg.fps,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
        )
    elif cfg.train_dataset_repo_id:
        # 기존 학습 데이터셋 메타 재사용 → 임시 폴더 생성 불필요
        dataset = LeRobotDataset(cfg.train_dataset_repo_id)
    else:
        import shutil
        tmp_repo = "data/tmp_eval_meta"
        tmp_path = Path.home() / ".cache/huggingface/lerobot/data/tmp_eval_meta"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        dataset = LeRobotDataset.create(
            tmp_repo,
            cfg.fps,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
        )

    # Policy 로딩
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    robot.connect()
    listener, events = init_keyboard_listener()

    # GPU 워밍업: 첫 에피소드 시작 전에 더미 inference로 CUDA 커널 초기화
    logging.info("GPU 워밍업 중...")
    _warmup_policy(robot, policy, preprocessor, postprocessor, dataset_features, cfg.task)
    logging.info("GPU 워밍업 완료.")

    # 첫 에피소드 시작 전에도 HOME으로 이동 → 정책 첫 액션과의 갭 제거
    _go_home(robot, cfg.fps)

    # 에피소드별 결과 추적
    results = []
    episode_idx = 0

    try:
        while episode_idx < cfg.num_episodes and not events["stop_recording"]:
            log_say(f"에피소드 {episode_idx + 1}/{cfg.num_episodes} 시작", cfg.play_sounds)
            logging.info(f"\n{'='*50}")
            logging.info(f"에피소드 {episode_idx + 1}/{cfg.num_episodes} 실행 중...")
            logging.info(f"태스크: {cfg.task}")
            logging.info("→ 키: 조기 종료 | ESC: 전체 중단")
            logging.info(f"{'='*50}")

            ep_result = run_eval_episode(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset_features=dataset_features,
                events=events,
                fps=cfg.fps,
                episode_time_s=cfg.episode_time_s,
                task=cfg.task,
                dataset=dataset if cfg.save_dataset else None,
            )

            logging.info("에피소드 종료. 로봇이 현재 위치를 유지합니다.")

            if events["stop_recording"]:
                if cfg.save_dataset:
                    dataset.clear_episode_buffer()
                break

            # 에러가 발생한 경우
            if ep_result["error"] is not None:
                logging.error(f"에피소드 에러 발생: {ep_result['error']}")
                logging.error("이 에피소드는 폐기됩니다.")
                if cfg.save_dataset:
                    dataset.clear_episode_buffer()
                results.append({"episode": episode_idx, "success": None, "steps": ep_result["num_steps"], "error": ep_result["error"]})
                episode_idx += 1
            else:
                logging.info(f"\n에피소드 {episode_idx + 1} 완료: {ep_result['num_steps']} 스텝")
                logging.info("성공 여부를 입력하세요: [s] 성공 / [d] 실패 / [ESC] 전체 종료")

                # 사용자 입력 대기
                success = _wait_for_episode_decision(events)

                if success is None or events["stop_recording"]:
                    if cfg.save_dataset:
                        dataset.clear_episode_buffer()
                    break

                # 성공/실패 모두 저장 (나중에 확인용). 결과는 results 리스트로만 구분.
                if cfg.save_dataset:
                    dataset.save_episode()
                    status = "성공" if success else "실패"
                    logging.info(f"에피소드 저장됨 ({status}).")

                results.append({"episode": episode_idx, "success": success, "steps": ep_result["num_steps"], "error": None})
                episode_idx += 1

            # 홈 포지션으로 복귀
            if not events["stop_recording"]:
                _go_home(robot, cfg.fps)

            # 마지막 에피소드가 아니면 리셋 대기
            if episode_idx < cfg.num_episodes and not events["stop_recording"]:
                should_continue = wait_for_reset(
                    robot=robot,
                    events=events,
                    fps=cfg.fps,
                    reset_time_s=cfg.reset_time_s,
                    task=cfg.task,
                )
                if not should_continue:
                    break

    finally:
        if robot.is_connected:
            robot.disconnect()
        if not is_headless() and listener:
            listener.stop()

        # 최종 결과 출력
        _print_summary(results)

        # 결과 JSON 저장
        _save_results_json(results, cfg)


def _warmup_policy(robot, policy, preprocessor, postprocessor, dataset_features, task):
    """
    첫 에피소드 전에 더미 inference를 1회 실행해 CUDA 커널을 초기화합니다.
    이렇게 하면 첫 스텝에서 발생하는 3~4 Hz 지연이 에피소드 실행 중이 아닌 시작 전에 발생합니다.
    """
    _, _, robot_observation_processor = make_default_processors()
    try:
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
    except Exception as e:
        logging.warning(f"워밍업 중 오류 (무시됨): {e}")


def _wait_for_episode_decision(events: dict, timeout_s: float = 30.0) -> bool | None:
    """
    사용자가 s(성공), d(폐기), ESC(종료)를 입력할 때까지 대기합니다.
    timeout_s 초 후에는 자동으로 폐기(False)를 반환합니다.

    pynput 없이도 동작하도록 이벤트 딕셔너리 기반으로 처리합니다.
    """
    if is_headless():
        logging.warning("Headless 환경: 에피소드 자동 폐기")
        return False

    from pynput import keyboard as kb

    decision = {"value": None}

    def on_press(key):
        try:
            if hasattr(key, "char"):
                if key.char == "s":
                    decision["value"] = True
                elif key.char == "d":
                    decision["value"] = False
            elif key == kb.Key.esc:
                events["stop_recording"] = True
                decision["value"] = None
        except Exception:
            pass

    listener = kb.Listener(on_press=on_press)
    listener.start()

    start_t = time.perf_counter()
    while decision["value"] is None and not events["stop_recording"]:
        if time.perf_counter() - start_t > timeout_s:
            logging.warning(f"{timeout_s:.0f}초 내 입력 없음. 자동 폐기.")
            decision["value"] = False
            break
        time.sleep(0.05)

    listener.stop()
    return decision["value"]


def _go_home(robot, fps: int, duration_s: float = 10.0):
    """현재 위치에서 홈 포지션까지 보간하며 천천히 이동."""
    HOME_POSITION = {
        "shoulder_pan.pos":   -8.24,
        "shoulder_lift.pos": -109.72,
        "elbow_flex.pos":    106.32,
        "wrist_flex.pos":     47.21,
        "wrist_roll.pos":     0,
        "gripper.pos":         1.45,
    }

    logging.info(f"홈 포지션으로 복귀 중... ({duration_s:.0f}초)")
    try:
        raw_obs = robot.get_observation()
        current = {k: v for k, v in raw_obs.items() if k.endswith(".pos")}
        steps = int(duration_s * fps)
        for i in range(1, steps + 1):
            t = i / steps
            interp = {k: current[k] + t * (HOME_POSITION[k] - current[k]) for k in HOME_POSITION}
            robot.send_action(interp)
            precise_sleep(1.0 / fps)
        logging.info("홈 포지션 도달.")
    except Exception as e:
        logging.warning(f"홈 복귀 실패 (무시): {e}")


def _print_summary(results: list[dict]):
    if not results:
        logging.info("\n평가된 에피소드가 없습니다.")
        return

    total = len(results)
    successes = sum(1 for r in results if r["success"] is True)
    failures = sum(1 for r in results if r["success"] is False)
    errors = sum(1 for r in results if r["error"] is not None)
    decided = total - errors

    logging.info("\n" + "=" * 50)
    logging.info("평가 결과 요약")
    logging.info("=" * 50)
    logging.info(f"총 에피소드:  {total}")
    logging.info(f"성공:         {successes}")
    logging.info(f"실패:         {failures}")
    logging.info(f"에러:         {errors}")
    if decided > 0:
        logging.info(f"성공률:       {successes / decided * 100:.1f}% ({successes}/{decided})")
    logging.info("=" * 50)

    for r in results:
        status = "✓ 성공" if r["success"] is True else ("✗ 실패" if r["success"] is False else f"! 에러: {r['error']}")
        logging.info(f"  에피소드 {r['episode'] + 1:2d}: {status} ({r['steps']} 스텝)")


def _save_results_json(results: list[dict], cfg: EvalRealRobotConfig):
    """평가 결과를 JSON 파일로 저장. 데이터셋 폴더 옆에 eval_results.json으로 떨어뜨림."""
    if not results:
        return

    import json
    from datetime import datetime

    total = len(results)
    successes = sum(1 for r in results if r["success"] is True)
    failures = sum(1 for r in results if r["success"] is False)
    errors = sum(1 for r in results if r["error"] is not None)
    decided = total - errors

    summary = {
        "timestamp": datetime.now().isoformat(),
        "policy_path": str(cfg.policy.pretrained_path) if cfg.policy else None,
        "dataset_repo_id": cfg.dataset_repo_id if cfg.save_dataset else None,
        "task": cfg.task,
        "num_episodes": total,
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "success_rate": successes / decided if decided > 0 else None,
        "episodes": results,
    }

    # 저장 위치: 데이터셋 저장된 곳 옆에 eval_results.json
    out_dir = Path.home() / ".cache/huggingface/lerobot" / cfg.dataset_repo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info(f"평가 결과 JSON 저장: {out_path}")


def main():
    init_logging()
    register_third_party_plugins()
    eval_real_robot()


if __name__ == "__main__":
    main()
