from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WINDOW_NAME = "PickLift v3 Collector"
START_RECT = (980, 480, 1260, 550)
STOP_RECT = (980, 570, 1260, 640)
QUIT_RECT = (980, 660, 1260, 730)


def _inside(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def render_dashboard(
    frame_rgb: np.ndarray,
    *,
    status: str,
    elapsed_s: float = 0,
    frames: int = 0,
    target_frames: int = 0,
    message: str = "",
    button_labels: tuple[str, str, str] = ("START", "END EPISODE", "QUIT"),
) -> np.ndarray:
    import cv2

    if frame_rgb.shape != (480, 640, 3):
        raise ValueError(f"expected RGB 640x480 frame, got {frame_rgb.shape}")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    camera = cv2.resize(frame_bgr, (960, 720), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((760, 1280, 3), 24, dtype=np.uint8)
    canvas[:720, :960] = camera
    cv2.rectangle(canvas, (0, 0), (960, 58), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "FRONT | aligned 1280x960 crop -> canonical 640x480 RGB",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
    )

    colors = {
        "WAITING": (0, 200, 255),
        "RECORDING": (40, 60, 255),
        "SAVING": (255, 180, 30),
        "SAVED": (60, 210, 80),
        "REVIEW": (190, 120, 255),
        "PRACTICE": (60, 210, 255),
        "STOPPED": (150, 150, 150),
        "ERROR": (30, 30, 230),
    }
    color = colors.get(status, (220, 220, 220))
    cv2.putText(canvas, status, (990, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)
    cv2.putText(
        canvas,
        f"Time   {elapsed_s:6.1f} s",
        (990, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
    )
    frame_text = f"Frames {frames:4d} / {target_frames:4d}" if target_frames else f"Frames {frames:6d}"
    cv2.putText(
        canvas,
        frame_text,
        (990, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        1,
    )
    instructions = (
        "Start: S / Enter / click",
        "End:   E / Space / click",
        "Quit:  Q / Esc / click",
    )
    for index, line in enumerate(instructions):
        cv2.putText(
            canvas,
            line,
            (990, 245 + index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
        )
    if message:
        for index, line in enumerate(message.splitlines()[:3]):
            cv2.putText(
                canvas,
                line[:34],
                (990, 380 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (160, 220, 255),
                1,
            )

    for rect, label, enabled_color in (
        (START_RECT, button_labels[0], (60, 170, 80)),
        (STOP_RECT, button_labels[1], (40, 80, 210)),
        (QUIT_RECT, button_labels[2], (90, 90, 90)),
    ):
        x0, y0, x1, y1 = rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), enabled_color, -1)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        cv2.putText(
            canvas,
            label,
            (x0 + (x1 - x0 - size[0]) // 2, y0 + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
    return canvas


@dataclass
class OperatorUI:
    target_frames: int
    _mouse_command: str | None = None

    def open(self) -> None:
        import cv2

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 760)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        import cv2

        if event != cv2.EVENT_LBUTTONUP:
            return
        if _inside(x, y, START_RECT):
            self._mouse_command = "start"
        elif _inside(x, y, STOP_RECT):
            self._mouse_command = "stop"
        elif _inside(x, y, QUIT_RECT):
            self._mouse_command = "quit"

    def show(
        self,
        frame_rgb: np.ndarray,
        *,
        status: str,
        elapsed_s: float = 0,
        frames: int = 0,
        message: str = "",
        button_labels: tuple[str, str, str] = ("START", "END EPISODE", "QUIT"),
    ) -> str | None:
        import cv2

        canvas = render_dashboard(
            frame_rgb,
            status=status,
            elapsed_s=elapsed_s,
            frames=frames,
            target_frames=self.target_frames,
            message=message,
            button_labels=button_labels,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        command = self._mouse_command
        self._mouse_command = None
        if key in (ord("s"), ord("S"), 13):
            return "start"
        if key in (ord("e"), ord("E"), 32):
            return "stop"
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key == ord("1"):
            return "start"
        if key == ord("2"):
            return "stop"
        if key == ord("3"):
            return "quit"
        return command

    def wait_for_start(self, frame_provider, message: str = "") -> None:
        while True:
            command = self.show(
                frame_provider(),
                status="WAITING",
                message=f"{message}\nConfirm setup, then START",
            )
            if command == "start":
                return
            if command == "quit":
                raise KeyboardInterrupt("operator quit before recording")
            time.sleep(0.01)

    def review_result(self, frame_rgb: np.ndarray) -> str:
        while True:
            command = self.show(
                frame_rgb,
                status="REVIEW",
                message="Choose result\n1 Success | 2 Failure | 3 Discard",
                button_labels=("SUCCESS", "FAILURE", "DISCARD"),
            )
            if command == "start":
                return "success"
            if command == "stop":
                return "failure"
            if command == "quit":
                return "discard"
            time.sleep(0.01)

    def show_complete(self, frame_rgb: np.ndarray, root: Path) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self.show(frame_rgb, status="SAVED", message=f"Saved:\n{root.name}")
            time.sleep(0.01)

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(WINDOW_NAME)
