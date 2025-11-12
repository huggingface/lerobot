#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Aria の Charuco/デモ分割結果を LeRobotDataset にポーティングするスクリプト。

要件:
- 1つめのスクリプト（Aria→Charuco/Segmentation）を呼び出して得られた "pairs" / "demonstrations" を
  LeRobotDataset へフレーム単位で格納する。
- フレームに入れるキーは以下のみ:
    is_first (bool[1])
    is_last (bool[1])
    is_terminal (bool[1])
    observation.state.camera_pose (float32[16])  # Charuco座標系 4x4 を row-major で16要素フラット化
    observation.images.active_vision (video: uint8[H,W,3])  # RGB（★回転後保存した整流画像を使用）
    action.active_vision (float32[16])  # 既定は observation と同じ。--action-next で次フレームの pose
"""

import argparse
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ----------------------------
# ユーティリティ
# ----------------------------

def flatten_T_rowmajor(T: np.ndarray) -> np.ndarray:
    """4x4 変換行列を row-major の 16要素 float32 にフラット化。"""
    T = np.asarray(T)
    if T.shape != (4, 4):
        raise ValueError(f"camera_pose must be 4x4, got {T.shape}")
    return T.astype(np.float32).reshape(-1)

def load_image_rgb(path: Path, resize_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """画像ファイルを RGB (H,W,3) uint8 でロード。必要なら (H,W) にリサイズ。"""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if resize_hw is not None:
        h, w = resize_hw
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb  # uint8

def infer_fps_from_timestamps_ns(ts_ns: List[int]) -> float:
    """タイムスタンプ列（ns）の差分から中央値をとり FPS 推定。"""
    ts = np.array(sorted(ts_ns), dtype=np.int64)
    if len(ts) < 2:
        return 30.0
    dts = np.diff(ts).astype(np.float64) * 1e-9  # to sec
    med = np.median(dts)
    if med <= 0:
        return 30.0
    return float(1.0 / med)

def dynamic_import_main_from_file(py_file: Path, symbol: str = "main") -> Callable:
    """py ファイルを動的 import して `symbol` を返す。"""
    spec = importlib.util.spec_from_file_location("aria_segment_runtime", str(py_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import spec from: {py_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    fn = getattr(mod, symbol, None)
    if fn is None:
        raise AttributeError(f"{py_file}: function `{symbol}` not found")
    return fn

# ----------------------------
# フレーム生成の肝
# ----------------------------

def generate_lerobot_frames(
    episode_pairs: List[Dict],
    resize_hw: Optional[Tuple[int, int]] = None,
    action_next: bool = False,
) -> Iterable[Dict[str, np.ndarray]]:
    """
    1エピソード（= 同じ demo_index のフレーム列）から LeRobot のフレーム辞書を順次 yield。

    必須キー（要求仕様）:
      - is_first: bool[1]
      - is_last: bool[1]
      - is_terminal: bool[1]
      - observation.state.camera_pose: float32[16]  # Charuco座標 4x4 を row-major 16
      - observation.images.active_vision: video (uint8[H,W,3])  # RGB
        ★ 前スクリプトで「保存時のみ90度回転」した整流画像（rgb_rect_path）を使用
      - action.active_vision: float32[16]
    """
    # 時系列順にソート
    episode_pairs = sorted(episode_pairs, key=lambda r: (r["rgb_timestamp_ns"], r["frame_index"]))

    # 事前に pose の None をチェック（Charuco 変換がない場合は不可）
    for r in episode_pairs:
        if r["pose_charuco_4x4"] is None:
            raise RuntimeError(
                "pose_charuco_4x4 is None. Charuco平均変換が求まっていないか、"
                "入力に問題があります（Charucoが全く検出できなかったなど）。"
            )

    # action_next=True の場合に備え、次フレームの pose を前計算
    pose_flat_list = [flatten_T_rowmajor(np.asarray(r["pose_charuco_4x4"])) for r in episode_pairs]
    next_pose_flat_list = pose_flat_list[1:] + [pose_flat_list[-1]]  # 最終フレームはそのまま引き継ぐ

    for idx, r in enumerate(episode_pairs):
        ### NEW/CHANGED ###
        # 画像は、回転後に保存された整流画像（rgb_rect_path）を使う
        img = load_image_rgb(Path(r["rgb_rect_path"]), resize_hw=resize_hw)

        obs_pose_flat = pose_flat_list[idx]
        act_pose_flat = next_pose_flat_list[idx] if action_next else obs_pose_flat

        frame = {
            "is_first": np.array([bool(r["is_first"])], dtype=bool),
            "is_last": np.array([bool(r["is_last"])], dtype=bool),
            "is_terminal": np.array([bool(r["is_terminal"])], dtype=bool),
            "observation.state.camera_pose": obs_pose_flat.astype(np.float32),
            "observation.images.active_vision": img,  # uint8 [H,W,3] RGB（回転後保存画像）
            "action.active_vision": act_pose_flat.astype(np.float32),
        }
        yield frame

# ----------------------------
# メイン処理
# ----------------------------

def build_features_for_image_shape(h: int, w: int) -> Dict:
    """最初の画像サイズに合わせて features を構築。"""
    return {
        "is_first": {"dtype": "bool", "shape": (1,), "names": None},
        "is_last": {"dtype": "bool", "shape": (1,), "names": None},
        "is_terminal": {"dtype": "bool", "shape": (1,), "names": None},
        "observation.state.camera_pose": {
            "dtype": "float32",
            "shape": (16,),
            "names": {"axes": [f"t{i:02d}" for i in range(16)]},
        },
        "observation.images.active_vision": {
            "dtype": "video",
            "shape": (h, w, 3),  # ★ 回転後保存画像の H,W に合わせる
            "names": ["height", "width", "channels"],
        },
        "action.active_vision": {
            "dtype": "float32",
            "shape": (16,),
            "names": {"axes": [f"t{i:02d}" for i in range(16)]},
        },
    }

def port_aria_to_lerobot(
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    # ArUco/Charuco/Segmentation params (1) スクリプトにそのまま渡す
    aruco_dict: str,
    squares_x: int,
    squares_y: int,
    square_len: float,
    marker_len: float,
    start_id: int,
    goal_id: int,
    start_hold_sec: float = 1.0,
    save_debug_overlay: bool = False,
    # 画像サイズオプション
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    # アクション仕様
    action_next: bool = False,
    # (1) スクリプト import 指定
    aria_module: Optional[str] = None,
    aria_script_file: Optional[Path] = None,
    # LeRobot
    push_to_hub: bool = False,
    robot_type: str = "ProjectAria",
):
    """
    (1) スクリプトの main() を呼び出して "pairs" と "demonstrations" を得て、
    LeRobotDataset に episode として保存する。
    """
    # 1) (1) スクリプトの main を import
    aria_main: Optional[Callable] = None

    # 優先順: 明示ファイル指定 > モジュール名指定 > 既定モジュール名
    if aria_script_file is not None:
        aria_main = dynamic_import_main_from_file(aria_script_file, "main")
    elif aria_module is not None:
        aria_main = importlib.import_module(aria_module).main  # type: ignore[attr-defined]
    else:
        try:
            from aria_charuco_segment import main as aria_main  # type: ignore
        except Exception as e:
            raise ImportError(
                "Aria用セグメンテーションスクリプトの import に失敗しました。"
                " --aria-module か --aria-script-file を指定してください。"
            ) from e

    # 2) (1) スクリプトを実行して分割 & Charuco変換
    vrs_path = input_dir / "sample.vrs"
    csv_path = input_dir / "slam" / "closed_loop_trajectory.csv"

    outputs = aria_main(
        str(vrs_path),
        str(csv_path),
        output_dir,
        squares_x=squares_x,
        squares_y=squares_y,
        square_len_m=square_len,
        marker_len_m=marker_len,
        aruco_name=aruco_dict,
        save_debug_overlay=save_debug_overlay,
        start_id=start_id,
        goal_id=goal_id,
        start_hold_sec=start_hold_sec,
        # 整流サイズ/回転は (1) 側のデフォルト or その CLI で制御
    )

    pairs: List[Dict] = outputs["pairs"]
    demonstrations: List[Dict] = outputs["demonstrations"]
    T_cw = outputs.get("T_charuco_world_avg", None)

    if T_cw is None:
        raise RuntimeError(
            "T_charuco_world_avg が None です。Charuco が全く検出できず平均変換が求められていません。"
            " Charucoが写るフレームを含めて再収集/再実行してください。"
        )

    # 3) エピソード（demo_index ごと）にグルーピング
    episodes_map: Dict[int, List[Dict]] = {}
    for r in pairs:
        di = r.get("demo_index", None)
        if di is None:
            continue  # デモ区間外はスキップ
        episodes_map.setdefault(int(di), []).append(r)

    if len(episodes_map) == 0:
        raise RuntimeError("デモエピソードが 0 件です。start/goal の検出設定を確認してください。")

    # 4) FPS 推定（全エピソードのタイムスタンプから中央値）
    all_ts = [r["rgb_timestamp_ns"] for rs in episodes_map.values() for r in rs]
    fps_est = infer_fps_from_timestamps_ns(all_ts)

    # 5) 最初の画像サイズ（またはリサイズ指定）から features 決定
    any_first_rec = sorted(episodes_map[min(episodes_map.keys())], key=lambda r: r["rgb_timestamp_ns"])[0]
    if resize_height is not None and resize_width is not None:
        h0, w0 = resize_height, resize_width
    else:
        ### NEW/CHANGED ###
        # features も回転後保存画像に合わせる（rgb_rect_path を参照）
        img0 = load_image_rgb(Path(any_first_rec["rgb_rect_path"]))
        h0, w0 = img0.shape[0], img0.shape[1]
    features = build_features_for_image_shape(h0, w0)

    # 6) LeRobotDataset を作成
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=float(fps_est),
        features=features,
    )
    logging.info(f"Create LeRobotDataset: repo_id={repo_id}, fps≈{fps_est:.3f}, image_shape=({h0},{w0},3)")

    # 7) エピソードごとにフレーム投入
    resize_hw = (resize_height, resize_width) if (resize_height is not None and resize_width is not None) else None
    total_frames = 0

    for ep_idx in sorted(episodes_map.keys()):
        ep_records = episodes_map[ep_idx]
        for frame in generate_lerobot_frames(ep_records, resize_hw=resize_hw, action_next=action_next):
            lerobot_dataset.add_frame(frame)
            total_frames += 1
        lerobot_dataset.save_episode()
        logging.info(f"Episode {ep_idx} saved: {len(ep_records)} frames")

    # 8) finalize & push (option)
    lerobot_dataset.finalize()
    logging.info(f"Finalize dataset. total_frames={total_frames}, episodes={len(episodes_map)}")

    if push_to_hub:
        lerobot_dataset.push_to_hub(tags=["openx"], private=False)
        logging.info("Pushed to hub.")

# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Project Aria → LeRobotDataset (Charuco座標のカメラpose + 画像) ポートツール")

    # 入出力
    parser.add_argument("--input-dir", type=Path, required=True, help="(1) スクリプトと同じ: sample.vrs / slam/closed_loop_trajectory.csv を含むディレクトリ")
    parser.add_argument("--output-dir", type=Path, required=True, help="(1) スクリプトが画像を書き出す先（既存指定でOK）")
    parser.add_argument("--repo-id", type=str, required=True, help="出力先 LeRobotDataset の repo_id")

    # Charuco / Segmentation
    parser.add_argument("--aruco-dict", type=str, default="4X4_50")
    parser.add_argument("--squares-x", type=int, required=True)
    parser.add_argument("--squares-y", type=int, required=True)
    parser.add_argument("--square-len", type=float, required=True, help="Charuco の正方形1辺[m]")
    parser.add_argument("--marker-len", type=float, required=True, help="Charuco のマーカー1辺[m]")
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--goal-id", type=int, required=True)
    parser.add_argument("--start-hold-sec", type=float, default=1.0)
    parser.add_argument("--save-debug-overlay", action="store_true")

    # 画像サイズ
    parser.add_argument("--resize-width", type=int, default=None, help="動画の横幅（指定時は全フレーム固定リサイズ）")
    parser.add_argument("--resize-height", type=int, default=None, help="動画の縦幅（指定時は全フレーム固定リサイズ）")

    # アクション設定
    parser.add_argument("--action-next", action="store_true", help="アクティブビジョンの action を「次フレームの pose」にする（未指定なら観測と同じ）")

    # (1) スクリプト import 指定
    parser.add_argument("--aria-module", type=str, default=None, help="(1) スクリプトを import するモジュール名（例: aria_charuco_segment）")
    parser.add_argument("--aria-script-file", type=Path, default=None, help="(1) スクリプトの .py ファイルパス（モジュールで渡せない場合）")

    # LeRobot
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--robot-type", type=str, default="ProjectAria")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    port_aria_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        aruco_dict=args.aruco_dict,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_len=args.square_len,
        marker_len=args.marker_len,
        start_id=args.start_id,
        goal_id=args.goal_id,
        start_hold_sec=args.start_hold_sec,
        save_debug_overlay=args.save_debug_overlay,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        action_next=args.action_next,
        aria_module=args.aria_module,
        aria_script_file=args.aria_script_file,
        push_to_hub=args.push_to_hub,
        robot_type=args.robot_type,
    )

if __name__ == "__main__":
    main()
