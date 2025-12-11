"""
リアルタイム軌跡オーバーレイ録画

ポリシー実行時の各フレームでactionから予測されるエンドエフェクタ位置を
計算し、軌跡として動画にオーバーレイして保存する。
"""
from __future__ import annotations

import atexit
import json
import logging
import platform
import signal
import weakref
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import torch

# グローバルなレコーダー参照リスト（終了時のクリーンアップ用）
_active_recorders: list[weakref.ref] = []
_cleanup_done = False


def _cleanup_all_recorders():
    """プログラム終了時に全てのレコーダーを閉じる"""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    for ref in _active_recorders:
        recorder = ref()
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
    _active_recorders.clear()


def _signal_handler(signum, frame):
    """シグナルハンドラ：Ctrl+C等でクリーンアップを実行"""
    _cleanup_all_recorders()
    signal.default_int_handler(signum, frame)


# atexitとシグナルハンドラの両方を登録
atexit.register(_cleanup_all_recorders)
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass


# URDFモジュールは遅延インポート
_URDF = None
_USE_URCHIN = False


def _get_urdf_class():
    """URDFクラスを遅延インポートして返す"""
    global _URDF, _USE_URCHIN
    if _URDF is not None:
        return _URDF

    try:
        from urchin import URDF
        _URDF = URDF
        _USE_URCHIN = True
        return _URDF
    except ImportError:
        pass

    try:
        from urdfpy import URDF
        _URDF = URDF
        _USE_URCHIN = False
        return _URDF
    except ImportError:
        raise ImportError(
            "URDF library not found. Please install 'urchin':\n"
            "  pip install urchin"
        )


def _get_available_codec() -> str:
    """クロスプラットフォームで利用可能なコーデックを検出"""
    preferred_codecs = ["libx264", "mpeg4"]
    if platform.system() == "Darwin":
        preferred_codecs.insert(0, "h264_videotoolbox")

    for codec in preferred_codecs:
        try:
            av.codec.Codec(codec, "w")
            return codec
        except Exception:
            continue
    return "mpeg4"


# SO-101の関節キー
ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _extract_first_image_from_obs_frame(observation_frame: dict[str, Any]) -> np.ndarray | None:
    """observation_frameから最初のBGR画像を抽出"""
    for key, value in observation_frame.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            continue
        if arr.ndim != 3:
            continue
        img = arr
        if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = np.ascontiguousarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return None


class TrajectoryVideoRecorder:
    """PyAVを使用したクロスプラットフォーム対応のビデオレコーダー"""

    def __init__(self, output_path: Path, fps: int):
        self.output_path = Path(output_path)
        self.fps = fps
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._frame_count = 0
        self._closed = False
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._codec = _get_available_codec()
        _active_recorders.append(weakref.ref(self))

    def _open_writer(self, w: int, h: int) -> None:
        self._container = av.open(str(self.output_path), "w")
        self._stream = self._container.add_stream(self._codec, rate=self.fps)
        self._stream.width = w
        self._stream.height = h
        self._stream.pix_fmt = "yuv420p"
        if self._codec in ("libx264", "h264_videotoolbox"):
            self._stream.options = {"crf": "23", "preset": "fast"}
        logging.info(f"TrajectoryVideoRecorder: using codec '{self._codec}' for {self.output_path}")

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self._container is None:
            self._open_writer(w, h)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        av_frame.pts = self._frame_count
        self._frame_count += 1

        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._container is not None:
            try:
                if self._stream is not None:
                    for packet in self._stream.encode():
                        self._container.mux(packet)
                self._container.close()
                logging.info(f"TrajectoryVideoRecorder: closed {self.output_path} ({self._frame_count} frames)")
            except Exception as e:
                logging.warning(f"TrajectoryVideoRecorder: error closing {self.output_path}: {e}")
            finally:
                self._container = None
                self._stream = None
                self._frame_count = 0

    def __del__(self):
        self.close()


class TrajectoryRecordingManager:
    """
    リアルタイム軌跡オーバーレイ録画マネージャー

    毎フレームの action から FK 計算を行い、
    エンドエフェクタの軌跡をカメラ画像にオーバーレイして動画保存する。
    """

    def __init__(
        self,
        output_root: Path,
        fps: int,
        urdf_path: Path,
        ee_link_name: str = "gripper_frame_link",
        affine_conf_path: Path | None = None,
        max_history_points: int = 300,  # 約10秒分 @30fps
    ):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.urdf_path = Path(urdf_path)
        self.ee_link_name = ee_link_name
        self.max_history_points = max_history_points

        # URDF & affine matrixのロード
        self._robot = None
        self._ee_link = None
        self._urdf_joint_names: list[str] = []
        self._affine: np.ndarray | None = None

        # affine_conf_path が None なら SO101 のデフォルトを使う
        if affine_conf_path is None:
            affine_conf_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "SO101" / "fk_image.conf"
        self._affine_conf_path = affine_conf_path

        # 状態
        self._writer: TrajectoryVideoRecorder | None = None
        self._episode_idx: int | None = None
        self._trajectory_uv: list[tuple[int, int]] = []  # 過去の軌跡点
        self._frame_idx: int = 0
        self._initialized = False

    def _lazy_init(self) -> bool:
        """URDFとaffineの遅延ロード"""
        if self._initialized:
            return self._robot is not None

        self._initialized = True

        # URDFロード
        if not self.urdf_path.exists():
            logging.warning(f"TrajectoryRecordingManager: URDF not found: {self.urdf_path}")
            return False

        try:
            URDF = _get_urdf_class()
            if _USE_URCHIN:
                self._robot = URDF.load(str(self.urdf_path), lazy_load_meshes=True)
            else:
                self._robot = URDF.load(str(self.urdf_path))

            movable_joints = [j for j in self._robot.joints if getattr(j, "joint_type", None) != "fixed"]
            self._urdf_joint_names = [j.name for j in movable_joints]

            if self.ee_link_name not in self._robot.link_map:
                logging.warning(f"TrajectoryRecordingManager: EE link '{self.ee_link_name}' not found in URDF")
                self._robot = None
                return False

            self._ee_link = self._robot.link_map[self.ee_link_name]
            logging.info(f"TrajectoryRecordingManager: URDF loaded ({len(self._urdf_joint_names)} joints)")

        except Exception as e:
            logging.warning(f"TrajectoryRecordingManager: Failed to load URDF: {e}")
            return False

        # Affine matrixロード
        if not self._affine_conf_path.exists():
            logging.warning(f"TrajectoryRecordingManager: fk_image.conf not found: {self._affine_conf_path}")
            return False

        try:
            with self._affine_conf_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._affine = np.asarray(data["affine_matrix"], dtype=float)
            if self._affine.shape != (2, 3):
                logging.warning(f"TrajectoryRecordingManager: Invalid affine matrix shape: {self._affine.shape}")
                self._affine = None
                return False
            logging.info("TrajectoryRecordingManager: affine matrix loaded")
        except Exception as e:
            logging.warning(f"TrajectoryRecordingManager: Failed to load affine matrix: {e}")
            return False

        return True

    def start_episode(self, episode_idx: int) -> None:
        """エピソード開始"""
        if not self._lazy_init():
            logging.warning(f"TrajectoryRecordingManager: _lazy_init failed, skipping episode {episode_idx}")
            return

        self._episode_idx = episode_idx
        video_name = f"trajectory_episode_{episode_idx}.mp4"
        self._writer = TrajectoryVideoRecorder(self.output_root / video_name, fps=self.fps)
        self._trajectory_uv = []
        self._frame_idx = 0
        logging.info(f"TrajectoryRecordingManager: started episode {episode_idx}")

    def _compute_fk(self, action: dict[str, Any]) -> np.ndarray | None:
        """単一のactionからEE位置を計算"""
        if self._robot is None or self._ee_link is None:
            return None

        try:
            joint_cfg: dict[str, float] = {}
            for key in ACTION_KEYS:
                if key not in action:
                    return None
                base_name = key.split(".")[0]
                val_deg = float(action[key])
                if base_name in self._urdf_joint_names:
                    joint_cfg[base_name] = float(np.deg2rad(val_deg))

            fk_all = self._robot.link_fk(joint_cfg)
            T = fk_all[self._ee_link]
            return np.asarray(T[:3, 3], dtype=float)
        except Exception:
            return None

    def _project_to_image(self, pos_3d: np.ndarray) -> tuple[int, int]:
        """3D位置を画像座標に投影"""
        x, y = float(pos_3d[0]), float(pos_3d[1])
        vec = np.array([x, y, 1.0], dtype=float)
        uv = self._affine @ vec
        return int(round(uv[0])), int(round(uv[1]))

    def log_frame(
        self,
        observation_frame: dict[str, Any],
        action_values: dict[str, Any],
    ) -> None:
        """
        フレームを処理して軌跡をオーバーレイ

        Args:
            observation_frame: observation辞書（画像を含む）
            action_values: 現在のaction（関節位置辞書）
        """
        if self._writer is None or self._robot is None or self._affine is None:
            if self._frame_idx == 0:
                logging.warning(f"TrajectoryRecordingManager: log_frame skipped (writer={self._writer is not None}, robot={self._robot is not None}, affine={self._affine is not None})")
            return

        frame_bgr = _extract_first_image_from_obs_frame(observation_frame)
        if frame_bgr is None:
            if self._frame_idx == 0:
                logging.warning("TrajectoryRecordingManager: no image found in observation_frame")
            return

        # 初回フレームでaction_valuesのキーをログ出力
        if self._frame_idx == 0:
            logging.info(f"TrajectoryRecordingManager: action_values keys = {list(action_values.keys())[:10]}")

        h, w = frame_bgr.shape[:2]
        output_frame = frame_bgr.copy()

        # 現在のactionからEE位置を計算
        pos_3d = self._compute_fk(action_values)
        if pos_3d is not None:
            uv = self._project_to_image(pos_3d)
            self._trajectory_uv.append(uv)

            # 履歴を制限
            if len(self._trajectory_uv) > self.max_history_points:
                self._trajectory_uv.pop(0)

        # 軌跡を描画（古いほど薄く）
        n_points = len(self._trajectory_uv)
        if n_points >= 2:
            for i in range(1, n_points):
                # alpha: 古いほど薄く
                alpha = 0.3 + 0.7 * (i / n_points)
                color = (0, 0, int(255 * alpha))  # 赤
                cv2.line(output_frame, self._trajectory_uv[i - 1], self._trajectory_uv[i], color, 2)

        # 現在位置を緑丸で表示
        if self._trajectory_uv:
            u_cur, v_cur = self._trajectory_uv[-1]
            cv2.circle(output_frame, (u_cur, v_cur), 6, (0, 255, 0), -1)

        # デバッグ情報
        info = f"frame={self._frame_idx} trajectory_pts={n_points}"
        cv2.putText(output_frame, info, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        self._writer.add_frame(output_frame)
        self._frame_idx += 1

    def finish_episode(self) -> None:
        """エピソード終了"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._trajectory_uv = []
        self._frame_idx = 0
        self._episode_idx = None

    def finalize(self) -> None:
        """全体の終了処理"""
        self.finish_episode()
