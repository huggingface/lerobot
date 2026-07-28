from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WINDOW_NAME = "PickLift v3 Collector"
START_RECT = (980, 480, 1260, 550)
STOP_RECT = (980, 570, 1260, 640)
QUIT_RECT = (980, 660, 1260, 730)
BUTTON_RECTS = (START_RECT, STOP_RECT, QUIT_RECT)
BUTTON_COMMANDS = ("start", "stop", "quit")


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
    buttons_enabled: tuple[bool, bool, bool] = (True, True, True),
    pressed_button: int | None = None,
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
        "READY": (0, 200, 255),
        "CONNECTING": (255, 180, 30),
        "ACCEPTED": (60, 210, 80),
        "NOT SAVED": (30, 140, 230),
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
    shortcut_text = ("S / Enter / click", "E / Space / click", "Q / Esc / click")
    instructions = [
        f"{label.split(' / ')[0][:12]}: {shortcut}"
        for label, shortcut in zip(button_labels, shortcut_text, strict=True)
        if label
    ]
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
        for index, line in enumerate(message.splitlines()[:5]):
            cv2.putText(
                canvas,
                line[:34],
                (990, 345 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (160, 220, 255),
                1,
            )

    enabled_colors = ((60, 170, 80), (40, 80, 210), (90, 90, 90))
    for index, (rect, label, enabled_color) in enumerate(
        zip(BUTTON_RECTS, button_labels, enabled_colors, strict=True)
    ):
        x0, y0, x1, y1 = rect
        fill = enabled_color if buttons_enabled[index] else (52, 52, 52)
        if pressed_button == index:
            fill = tuple(min(channel + 55, 255) for channel in enabled_color)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
        if pressed_button == index:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 4)
        if not label:
            continue
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
    feedback_seconds: float = 0.18
    debounce_seconds: float = 0.45
    _input_lock_until: float = field(default=0, init=False, repr=False)

    def open(self) -> None:
        import cv2

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 760)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        import cv2

        if event != cv2.EVENT_LBUTTONUP:
            return
        for rect, command in zip(BUTTON_RECTS, BUTTON_COMMANDS, strict=True):
            if _inside(x, y, rect):
                self._mouse_command = command
                return

    @staticmethod
    def _command_index(command: str | None) -> int | None:
        try:
            return BUTTON_COMMANDS.index(command)
        except ValueError:
            return None

    def _acknowledge(
        self,
        frame_rgb: np.ndarray,
        *,
        command: str,
        button_labels: tuple[str, str, str],
        elapsed_s: float,
        frames: int,
    ) -> None:
        import cv2

        index = self._command_index(command)
        if index is None:
            return
        self._input_lock_until = time.monotonic() + self.debounce_seconds
        label = button_labels[index] or command.upper()
        labels = list(button_labels)
        labels[index] = f"{label} ..."
        canvas = render_dashboard(
            frame_rgb,
            status="ACCEPTED",
            elapsed_s=elapsed_s,
            frames=frames,
            target_frames=self.target_frames,
            message=f"{label} accepted\nPlease wait; buttons locked",
            button_labels=tuple(labels),
            buttons_enabled=(False, False, False),
            pressed_button=index,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        deadline = time.monotonic() + self.feedback_seconds
        while time.monotonic() < deadline:
            cv2.waitKey(1)
            self._mouse_command = None
            time.sleep(0.01)

    def show(
        self,
        frame_rgb: np.ndarray,
        *,
        status: str,
        elapsed_s: float = 0,
        frames: int = 0,
        message: str = "",
        button_labels: tuple[str, str, str] = ("START", "END EPISODE", "QUIT"),
        buttons_enabled: tuple[bool, bool, bool] = (True, True, True),
        acknowledge: bool = True,
    ) -> str | None:
        import cv2

        input_locked = time.monotonic() < self._input_lock_until
        effective_buttons_enabled = (False, False, False) if input_locked else buttons_enabled
        canvas = render_dashboard(
            frame_rgb,
            status=status,
            elapsed_s=elapsed_s,
            frames=frames,
            target_frames=self.target_frames,
            message=message,
            button_labels=button_labels,
            buttons_enabled=effective_buttons_enabled,
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        command = self._mouse_command
        self._mouse_command = None
        keyboard_command = None
        if key in (ord("s"), ord("S"), 13, ord("1")):
            keyboard_command = "start"
        elif key in (ord("e"), ord("E"), 32, ord("2")):
            keyboard_command = "stop"
        elif key in (ord("q"), ord("Q"), 27, ord("3")):
            keyboard_command = "quit"
        command = keyboard_command or command
        index = self._command_index(command)
        if index is None or input_locked or not buttons_enabled[index]:
            return None
        if acknowledge:
            self._acknowledge(
                frame_rgb,
                command=command,
                button_labels=button_labels,
                elapsed_s=elapsed_s,
                frames=frames,
            )
        return command

    def show_status(
        self,
        frame_rgb: np.ndarray,
        *,
        status: str,
        message: str,
        elapsed_s: float = 0,
        frames: int = 0,
    ) -> None:
        self.show(
            frame_rgb,
            status=status,
            elapsed_s=elapsed_s,
            frames=frames,
            message=message,
            button_labels=("", "", ""),
            buttons_enabled=(False, False, False),
            acknowledge=False,
        )

    def wait_for_start(self, frame_provider, message: str = "") -> None:
        while True:
            command = self.show(
                frame_provider(),
                status="WAITING",
                message=f"{message}\nConfirm setup, then START",
                buttons_enabled=(True, False, True),
            )
            if command == "start":
                return
            if command == "quit":
                raise KeyboardInterrupt("operator quit before recording")
            time.sleep(0.01)

    def wait_for_ready(self, frame_rgb: np.ndarray, message: str) -> bool:
        while True:
            command = self.show(
                frame_rgb,
                status="READY",
                message=message,
                button_labels=("READY / CONNECT", "", "QUIT"),
                buttons_enabled=(True, False, True),
            )
            if command == "start":
                return True
            if command == "quit":
                return False
            time.sleep(0.01)

    def review_result(self, frame_rgb: np.ndarray) -> str:
        while True:
            command = self.show(
                frame_rgb,
                status="REVIEW",
                message=(
                    "Choose result (manual visual)\n"
                    "FAILURE = task criteria unmet\n"
                    "DISCARD = record/config/safety issue"
                ),
                button_labels=("SUCCESS", "FAILURE", "DISCARD"),
            )
            if command == "start":
                while True:
                    confirmation = self.show(
                        frame_rgb,
                        status="REVIEW",
                        message=(
                            "Confirm SUCCESS: lift >=5cm\n"
                            "Between both fingers; no support\n"
                            "Held >=0.5s and still held at END"
                        ),
                        button_labels=("CONFIRM", "BACK", "DISCARD"),
                    )
                    if confirmation == "start":
                        return "success"
                    if confirmation == "stop":
                        break
                    if confirmation == "quit":
                        return "discard"
                    time.sleep(0.01)
            if command == "stop":
                return "failure"
            if command == "quit":
                return "discard"
            time.sleep(0.01)

    def show_complete(self, frame_rgb: np.ndarray, root: Path) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self.show(
                frame_rgb,
                status="SAVED",
                message=f"Saved:\n{root.name}",
                button_labels=("SAVED", "", ""),
                buttons_enabled=(False, False, False),
                acknowledge=False,
            )
            time.sleep(0.01)

    def show_saving(self, frame_rgb: np.ndarray, *, result: str) -> None:
        self.show_status(
            frame_rgb,
            status="SAVING",
            message=f"{result.upper()} accepted\nEncoding and writing; please wait",
        )

    def show_attempt_complete(
        self,
        frame_rgb: np.ndarray,
        *,
        result: str,
        saved_to_training: bool,
        next_message: str,
    ) -> None:
        deadline = time.monotonic() + 1
        status = "SAVED" if saved_to_training else "NOT SAVED"
        while time.monotonic() < deadline:
            self.show(
                frame_rgb,
                status=status,
                message=f"{result.upper()}\n{next_message}",
                button_labels=("SAVED" if saved_to_training else "RETRY", "", ""),
                buttons_enabled=(False, False, False),
                acknowledge=False,
            )
            time.sleep(0.01)

    def show_connection_error(self, frame_rgb: np.ndarray) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self.show(
                frame_rgb,
                status="ERROR",
                message=(
                    "Connection failed; no data written\nCheck arm power/cable\nReturning to READY for retry"
                ),
                button_labels=("", "", ""),
                buttons_enabled=(False, False, False),
                acknowledge=False,
            )
            time.sleep(0.01)

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(WINDOW_NAME)
