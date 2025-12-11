#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import cv2
import numpy as np

# --- NumPy 2.x 互換パッチ: urdfpy / urchin が np.float を使うのでエイリアスを貼る ---
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

# URDF モジュールは遅延インポート（オプション依存）
_URDF = None
_USE_URCHIN = False


def _get_urdf_class():
    """URDFクラスを遅延インポートして返す"""
    global _URDF, _USE_URCHIN
    if _URDF is not None:
        return _URDF

    # まず urchin を優先的に使う（あればメッシュ lazy ロード可能）
    try:
        from urchin import URDF  # type: ignore
        _URDF = URDF
        _USE_URCHIN = True
        return _URDF
    except ImportError:
        pass

    try:
        from urdfpy import URDF  # type: ignore
        _URDF = URDF
        _USE_URCHIN = False
        return _URDF
    except ImportError:
        raise ImportError(
            "URDF library not found. Please install either 'urchin' or 'urdfpy':\n"
            "  pip install urchin\n"
            "  or\n"
            "  pip install urdfpy"
        )


# ログに出てくる SO-101 の関節キー（LeRobot の action dict）
ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize VLA planned trajectory (future horizon) on attention video. "
            "For each chunk (frame_idx), actions[] are treated as the future plan."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help=(
            "Root directory of the recorded dataset, i.e. the parent of 'attn_videos'. "
            "Example: ./dataset/eval_test など。"
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index used in episode_values_{episode}.json (0-based).",
    )

    # URDF はこのスクリプトと同じ scripts フォルダに置く前提
    script_dir = Path(__file__).resolve().parent
    default_urdf_path = script_dir / "SO101" / "so101_new_calib.urdf"

    parser.add_argument(
        "--urdf-path",
        type=str,
        default=str(default_urdf_path),
        help=(
            "Path to SO-101 URDF file "
            "(default: SO101/so101_new_calib.urdf next to this script)."
        ),
    )
    parser.add_argument(
        "--ee-link",
        type=str,
        default="gripper_frame_link",
        help="End-effector link name in the URDF (default: gripper_frame_link).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="plan_overlay",
        help=(
            "Suffix for output video file. "
            "Output path will be attn_videos/{suffix}_episode_{episode}.mp4 "
            "(default: plan_overlay)."
        ),
    )
    return parser.parse_args()


# ====== STEP 1: JSON ログ読み込み ======

def load_episode_log(attn_dir: Path, episode: int) -> Dict[str, Any]:
    """
    {dataset_root}/attn_videos/episode_values_{episode}.json を読む。
    """
    json_path = attn_dir / f"episode_values_{episode}.json"
    print(f"[STEP 1] Loading JSON log: {json_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON log file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError(f"Invalid log format: 'frames' missing or not a list in {json_path}")

    total_actions = 0
    for rec in data["frames"]:
        actions = rec.get("actions", [])
        if isinstance(actions, list):
            total_actions += len(actions)

    print(
        f"[STEP 1] OK: loaded {len(data['frames'])} chunks, "
        f"total {total_actions} actions in episode {episode}"
    )
    return data


# ====== STEP 2: 動画を開く ======

def resolve_video_path(attn_dir: Path, episode: int, repo_id: str | None) -> Path:
    """
    実際のファイル名が不明なので、複数パターンを試す。
    優先順位:
      1) attn_all_cameras_episode_{episode}.mp4
      2) {repo_id.replace('/', '_')}_ep{episode:06d}.mp4
    どれも無かったら attn_videos の中身を全部列挙してからエラー。
    """
    candidates: List[Path] = []
    candidates.append(attn_dir / f"attn_all_cameras_episode_{episode}.mp4")

    if repo_id is not None:
        repo_sanitized = repo_id.replace("/", "_")
        candidates.append(attn_dir / f"{repo_sanitized}_ep{episode:06d}.mp4")

    print("[STEP 2] Resolving attention video path.")
    for p in candidates:
        print(f"[STEP 2]  Trying candidate: {p}")
        if p.exists():
            print(f"[STEP 2]  -> FOUND: {p}")
            return p

    print("[STEP 2] ERROR: none of the candidate videos exist.")
    if attn_dir.exists():
        print("[STEP 2] Existing files in attn_videos:")
        for x in sorted(attn_dir.iterdir()):
            print(f"  - {x.name}")
    else:
        print(f"[STEP 2] attn_videos directory does not exist at: {attn_dir}")

    raise FileNotFoundError(
        f"No attention video found for episode {episode} in {attn_dir}. "
        f"Check the actual file names listed above."
    )


def open_video(attn_dir: Path, episode: int, repo_id: str | None) -> cv2.VideoCapture:
    video_path = resolve_video_path(attn_dir, episode, repo_id)
    print(f"[STEP 2] Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"[STEP 2] OK: video opened successfully "
        f"(frames={frame_count}, fps={fps:.2f}, size={width}x{height})"
    )
    return cap


# ====== STEP 3: 各 chunk の「未来軌跡 (plan)」を FK で計算 ======

def build_joint_config_from_action(
    action: Dict[str, Any],
    urdf_joint_names: List[str],
) -> Dict[str, float]:
    if len(ACTION_KEYS) > len(urdf_joint_names):
        raise ValueError(
            f"Number of ACTION_KEYS ({len(ACTION_KEYS)}) is greater than number of URDF joints "
            f"({len(urdf_joint_names)}). Cannot build joint config."
        )

    # "shoulder_pan.pos" -> "shoulder_pan"
    base_vals_deg: Dict[str, float] = {}
    for key in ACTION_KEYS:
        if key not in action:
            raise KeyError(f"Action key '{key}' not found in action dict: {list(action.keys())}")
        base_name = key.split(".")[0]
        base_vals_deg[base_name] = float(action[key])

    joint_cfg: Dict[str, float] = {}
    for joint_name in urdf_joint_names:
        if joint_name not in base_vals_deg:
            continue
        joint_cfg[joint_name] = float(np.deg2rad(base_vals_deg[joint_name]))
    return joint_cfg


def compute_fk_plans(
    urdf_path: Path,
    ee_link_name: str,
    frames: List[Dict[str, Any]],
) -> Tuple[List[int], List[List[np.ndarray]]]:
    """
    episode_values_*.json の frames から、
      - chunk_starts: 各 chunk の frame_idx のリスト
      - plans_3d: 各 chunk ごとの「未来軌跡」の 3D 位置リスト (actions の長さ分)
    を返す。

    ここで actions は「n_action_steps step horizon の joint 指令列」とみなす。
    """
    print(f"[STEP 3] Loading URDF: {urdf_path}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # 遅延インポートでURDFクラスを取得
    URDF = _get_urdf_class()
    if _USE_URCHIN:
        robot = URDF.load(str(urdf_path), lazy_load_meshes=True)
    else:
        robot = URDF.load(str(urdf_path))

    print(f"[STEP 3] URDF loaded. Links: {len(robot.link_map)}, Joints: {len(robot.joint_map)}")

    movable_joints = [j for j in robot.joints if getattr(j, "joint_type", None) != "fixed"]
    urdf_joint_names = [j.name for j in movable_joints]
    print(f"[STEP 3] Movable joints ({len(urdf_joint_names)}): {urdf_joint_names}")

    if len(urdf_joint_names) < len(ACTION_KEYS):
        raise ValueError(
            f"URDF has fewer movable joints ({len(urdf_joint_names)}) than ACTION_KEYS ({len(ACTION_KEYS)})."
        )

    if ee_link_name not in robot.link_map:
        raise ValueError(
            f"EE link '{ee_link_name}' not found in URDF. "
            f"Available links example: {list(robot.link_map.keys())[:10]}"
        )
    ee_link = robot.link_map[ee_link_name]

    # 念のため frame_idx でソート
    records = sorted(frames, key=lambda r: int(r["frame_idx"]))

    chunk_starts: List[int] = []
    plans_3d: List[List[np.ndarray]] = []

    total_actions = 0

    for rec in records:
        base_idx = int(rec["frame_idx"])
        actions = rec.get("actions", [])
        if not isinstance(actions, list) or len(actions) == 0:
            continue

        plan_points: List[np.ndarray] = []
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError(f"Action must be a dict, got {type(action)}")
            joint_cfg = build_joint_config_from_action(action, urdf_joint_names)
            fk_all = robot.link_fk(joint_cfg)
            T = fk_all[ee_link]
            pos = np.asarray(T[:3, 3], dtype=float)
            plan_points.append(pos)
            total_actions += 1

        chunk_starts.append(base_idx)
        plans_3d.append(plan_points)

    print(
        f"[STEP 3] OK: computed plans for {len(chunk_starts)} chunks "
        f"(total actions processed={total_actions})."
    )
    if chunk_starts:
        print("[STEP 3] Sample plan (first chunk, first 5 points):")
        for i, pos in enumerate(plans_3d[0][:5]):
            print(f"  step {i}: EE pos = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    return chunk_starts, plans_3d


# ====== STEP 4: fk_image.conf の読み込み ======

def load_fk_image_conf(attn_dir: Path | None = None) -> np.ndarray:
    """
    scripts/SO101/fk_image.conf から affine_matrix を読む。
    attn_dir 引数は後方互換のために受け取るだけで、探索は SO101 固定。
    """
    script_dir = Path(__file__).resolve().parent
    conf_path = script_dir / "SO101" / "fk_image.conf"
    print("[STEP 4] Loading calibration config (SO101).")
    print(f"[STEP 4]  Path: {conf_path}")
    if not conf_path.exists():
        raise FileNotFoundError(
            "Calibration file fk_image.conf not found.\n"
            "Run fk_image_calibration.py to generate it."
        )
    print(f"[STEP 4]  -> FOUND: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "affine_matrix" not in data:
        raise ValueError("fk_image.conf must contain key 'affine_matrix'.")

    A = np.asarray(data["affine_matrix"], dtype=float)
    if A.shape != (2, 3):
        raise ValueError(
            f"fk_image.conf['affine_matrix'] must be shape (2, 3), got {A.shape} instead."
        )

    print("[STEP 4] OK: loaded affine_matrix from fk_image.conf:")
    print(A)
    return A

def project_world_to_image(affine: np.ndarray, pos_3d: np.ndarray) -> tuple[int, int]:
    x, y = float(pos_3d[0]), float(pos_3d[1])
    vec = np.array([x, y, 1.0], dtype=float)
    uv = affine @ vec  # shape (2,)
    u, v = int(round(uv[0])), int(round(uv[1]))
    return u, v


# ====== STEP 5: 「計画軌跡」をオーバーレイして動画書き出し ======

def overlay_plan_trajectory_video(
    cap: cv2.VideoCapture,
    chunk_starts: List[int],
    plans_3d: List[List[np.ndarray]],
    affine: np.ndarray,
    output_path: Path,
) -> None:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Discord/Notion で再生されやすいコーデックを優先
    writer = None
    for fourcc_tag in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    print(f"[STEP 5] Writing overlay video to: {output_path}")
    print(f"[STEP 5]  frames={frame_count}, fps={fps:.2f}, size={width}x{height}")

    # あらかじめ各 plan の 2D 投影を計算しておく
    plans_uv: List[List[Tuple[int, int]]] = []
    for plan in plans_3d:
        uv_list = [project_world_to_image(affine, p) for p in plan]
        plans_uv.append(uv_list)

    # 各チャンクの計画軌跡をオーバーレイ画像として事前生成（赤線）
    plan_overlays: List[np.ndarray] = []
    for plan_uv in plans_uv:
        overlay = np.zeros((height, width, 3), dtype=np.float32)
        if len(plan_uv) >= 2:
            for i in range(1, len(plan_uv)):
                cv2.line(overlay, plan_uv[i - 1], plan_uv[i], (0, 0, 255), 2)
        plan_overlays.append(overlay)

    # chunk_starts は昇順前提
    num_chunks = len(chunk_starts)
    if num_chunks == 0:
        print("[STEP 5] WARNING: no chunks to visualize (no plans).")
        # 単に元動画をコピーして終了
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        return

    # 古い軌跡の透明度が動画末尾でもうっすら残るように減衰係数を設定
    base_alpha = 0.9
    min_alpha = 0.12  # 最古チャンクの目標透明度
    if num_chunks > 1:
        decay = (min_alpha / base_alpha) ** (1 / max(num_chunks - 1, 1))
    else:
        decay = 1.0
    print(f"[STEP 5] alpha decay per chunk: base={base_alpha}, decay={decay:.4f}, min~{min_alpha}")

    current_chunk_idx = 0

    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            print(f"[STEP 5] WARNING: failed to read frame {frame_idx}, stopping.")
            break

        # 現在 frame に対応する chunk を更新
        while (
            current_chunk_idx + 1 < num_chunks
            and frame_idx >= chunk_starts[current_chunk_idx + 1]
        ):
            current_chunk_idx += 1

        # frame が最初の chunk 以前なら何も描かない
        if frame_idx < chunk_starts[0]:
            writer.write(frame)
            continue

        plan_uv = plans_uv[current_chunk_idx]
        if len(plan_uv) < 2:
            writer.write(frame)
            continue

        # 過去チャンクの軌跡も含めて重ね描き（古いほど薄く）
        overlay_acc = np.zeros_like(frame, dtype=np.float32)
        for past_idx in range(current_chunk_idx + 1):
            age = current_chunk_idx - past_idx
            alpha = base_alpha * (decay ** age)
            overlay_acc += plan_overlays[past_idx] * alpha
        overlay_uint8 = np.clip(overlay_acc, 0, 255).astype(np.uint8)
        frame = cv2.addWeighted(frame, 1.0, overlay_uint8, 1.0, 0.0)

        # 「今どこまで来ているか」を計画上で示す
        # t0 = chunk_starts[current_chunk_idx]
        # step_idx = clamp(frame_idx - t0, 0, len(plan_uv)-1)
        t0 = chunk_starts[current_chunk_idx]
        step_idx = frame_idx - t0
        if step_idx < 0:
            step_idx = 0
        if step_idx >= len(plan_uv):
            step_idx = len(plan_uv) - 1

        u_cur, v_cur = plan_uv[step_idx]
        cv2.circle(frame, (u_cur, v_cur), 5, (0, 255, 0), -1)

        # デバッグ用に現在の chunk / step 情報を表示
        info = f"chunk {current_chunk_idx}  frame {frame_idx}  plan_step {step_idx}/{len(plan_uv)-1}"
        cv2.putText(
            frame,
            info,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print("[STEP 5] Done writing overlay video.")


# ====== main ======

def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    episode = args.episode
    urdf_path = Path(args.urdf_path)
    ee_link_name = args.ee_link
    output_suffix = args.output_suffix

    print("=== visualize_inference_trajectory (plan horizon): start ===")
    print(f"  dataset-root = {dataset_root}")
    print(f"  episode      = {episode}")
    print(f"  urdf-path    = {urdf_path}")
    print(f"  ee-link      = {ee_link_name}")
    print(f"  using urchin = {_USE_URCHIN}")
    print(f"  output-suffix= {output_suffix}")

    attn_dir = dataset_root / "attn_videos"
    if not attn_dir.exists():
        raise FileNotFoundError(f"Attention directory not found: {attn_dir}")

    # 1. JSON ログ読み込み
    log = load_episode_log(attn_dir, episode)
    frames = log["frames"]
    repo_id = log.get("repo_id", None)
    if repo_id is not None:
        print(f"[INFO] repo_id from JSON = {repo_id}")

    # 2. 動画オープン
    cap = open_video(attn_dir, episode, repo_id)

    # 3. 各 chunk の未来軌跡 (plan) を FK で計算
    chunk_starts, plans_3d = compute_fk_plans(
        urdf_path=urdf_path,
        ee_link_name=ee_link_name,
        frames=frames,
    )

    # 4. キャリブレーション fk_image.conf 読み込み
    affine = load_fk_image_conf(attn_dir)

    # 5. 計画軌跡オーバーレイ動画の書き出し
    output_path = attn_dir / f"{output_suffix}_episode_{episode}.mp4"
    overlay_plan_trajectory_video(
        cap=cap,
        chunk_starts=chunk_starts,
        plans_3d=plans_3d,
        affine=affine,
        output_path=output_path,
    )

    print("=== visualize_inference_trajectory (plan horizon): done (step1-5 OK) ===")


if __name__ == "__main__":
    main()
