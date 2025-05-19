#!/usr/bin/env python
"""
Hand + Arm slider GUI for HopeJr with optional --no-hand / --no-arm flags.

Run examples:
    python sliders.py                 # hand + arm
    python sliders.py --no-hand       # arm only
    python sliders.py --no-arm        # hand only
"""

# ruff: noqa: N806

import json
import sys
from pathlib import Path

import numpy as np
import pygame

from lerobot.common.robots.hope_jr import HopeJrHand, HopeJrHandConfig

from .motors_bus import MotorsBus


# ────────────────────────────────────────────────────────────────────────────────
#  Helpers for JSON range files
# ────────────────────────────────────────────────────────────────────────────────
def load_ranges(path: str):
    d = json.loads(Path(path).read_text())
    return d["start_pos"], d["end_pos"]


def save_ranges(path: str, lo, hi):
    d = {
        "start_pos": lo,
        "end_pos": hi,
        "homing_offset": [],
        "drive_mode": [],
        "calib_mode": "LINEAR",
    }
    Path(path).write_text(json.dumps(d, indent=2))


def main(bus: MotorsBus, motors: list[str] | None = None):
    motors = motors if motors else list(bus.motors)

    if not bus.is_connected:
        bus.connect()

    H_RANGE = "examples/hopejr/settings/hand_ranges.json"
    hand_lo, hand_hi = load_ranges(H_RANGE) if motors else ([], [])
    # arm_lo, arm_hi = load_ranges(A_RANGE) if arm_names else ([], [])

    pygame.init()
    screen = pygame.display.set_mode((1080, 720))
    pygame.display.set_caption("Hand + Arm Sliders")
    font = pygame.font.SysFont(None, 20)
    big = pygame.font.SysFont(None, 24)

    ROWS = len(motors)
    TOP, BOT = 100, 20
    ROW_H = (720 - TOP - BOT) // max(1, ROWS)
    BAR_H = ROW_H - 25

    HX, H_W = None, 0

    H_MAX = 1024
    h_vals = np.zeros(len(motors), int)

    BTN_W, BTN_H = 120, 28
    btns = [
        {
            "rect": pygame.Rect(HX, 20, BTN_W, BTN_H),
            "txt": "Hand  LOWER",
            "col": "hand",
            "type": "low",
        },
        {
            "rect": pygame.Rect(HX + BTN_W + 10, 20, BTN_W, BTN_H),
            "txt": "Hand  UPPER",
            "col": "hand",
            "type": "high",
        },
    ]
    edit_mode = None  # None or ("hand"/"arm","low"/"high")
    toast = None
    toast_timer = 0

    def bar(x, y, w, val, lo, hi, vmax, label):
        lo, hi = sorted((lo, hi))
        px_lo = int(w * lo / vmax)
        px_hi = int(w * hi / vmax)
        px_val = int(w * val / vmax)
        bar_y = y + 5
        pygame.draw.rect(screen, (200, 200, 200), (x, bar_y, w, BAR_H))
        pygame.draw.rect(screen, (80, 200, 80), (x, bar_y, px_val, BAR_H))
        pygame.draw.rect(screen, (255, 60, 60), (x, bar_y, px_lo, BAR_H))
        pygame.draw.rect(screen, (255, 60, 60), (x + px_hi, bar_y, w - px_hi, BAR_H))
        name = font.render(label, True, (255, 255, 255))
        screen.blit(name, name.get_rect(midbottom=(x + w // 2, y - 3)))
        screen.blit(font.render(str(lo), True, (80, 120, 255)), (x - 45, bar_y + BAR_H // 4))
        screen.blit(font.render(str(hi), True, (80, 120, 255)), (x + w - 55, bar_y + BAR_H // 4))
        screen.blit(font.render(str(val), True, (255, 255, 255)), (x + w + 10, bar_y + BAR_H // 4))

    def slider_at(pos):
        x, y = pos
        row = (y - TOP) // ROW_H
        if not 0 <= row < ROWS:
            return None
        y0 = TOP + row * ROW_H
        if not y0 <= y <= y0 + BAR_H + 5:
            return None
        if HX <= x <= HX + H_W and row < len(motors):
            return (str(bus), row, HX, H_W, H_MAX)
        return None

    def toast_msg(txt):
        nonlocal toast, toast_timer
        toast = big.render(txt, True, (255, 255, 0))
        toast_timer = 120

    clock = pygame.time.Clock()
    dragging = None
    while True:
        screen.fill((20, 20, 20))

        for i, n in enumerate(motors):
            bar(HX, TOP + i * ROW_H, H_W, h_vals[i], hand_lo[i], hand_hi[i], H_MAX, n)

        # for i, n in enumerate(arm_names):
        #     bar(AX, TOP + i * ROW_H, A_W, a_vals[i], arm_lo[i], arm_hi[i], A_MAX, n)

        # buttons
        for b in btns:
            col = (120, 120, 120) if edit_mode == (b["col"], b["type"]) else (60, 60, 60)
            pygame.draw.rect(screen, col, b["rect"])
            txt = font.render(b["txt"], True, (255, 255, 255))
            screen.blit(txt, txt.get_rect(center=b["rect"].center))

        if toast:
            screen.blit(toast, toast.get_rect(center=(540, 55)))

        pygame.display.flip()

        # events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                bus.disconnect()
                pygame.quit()
                sys.exit()

            if ev.type == pygame.MOUSEBUTTONDOWN:
                # button toggle
                for b in btns:
                    if b["rect"].collidepoint(ev.pos):
                        edit_mode = None if edit_mode == (b["col"], b["type"]) else (b["col"], b["type"])
                        toast_msg(f"Editing: {edit_mode[0]} {edit_mode[1]}" if edit_mode else "Edit off")
                        break
                else:
                    hit = slider_at(ev.pos)
                    if hit:
                        dragging = hit
                    # bound edit
                    if edit_mode and hit and hit[0] == edit_mode[0]:
                        col, row, left, w, vmax = hit
                        mx, _ = ev.pos
                        new = int(np.clip((mx - left) / w * vmax, 0, vmax))
                        if col == "hand":
                            if edit_mode[1] == "low":
                                hand_lo[row] = new
                            else:
                                hand_hi[row] = new
                            save_ranges(H_RANGE, hand_lo, hand_hi)
                        else:
                            raise ValueError(col)
                            # if edit_mode[1] == "low":
                            #     arm_lo[row] = new
                            # else:
                            #     arm_hi[row] = new
                            # save_ranges(A_RANGE, arm_lo, arm_hi)
                        toast_msg(f"{col} {row} {edit_mode[1]}→{new}")

            if ev.type == pygame.MOUSEBUTTONUP:
                dragging = None
            if ev.type == pygame.MOUSEMOTION and dragging:
                col, row, left, w, vmax = dragging
                mx, _ = ev.pos
                new = int(np.clip((mx - left) / w * vmax, 0, vmax))
                if h_vals[row] != new:
                    h_vals[row] = new
                    bus.write("Goal_Position", [new], [motors[row]])

        if toast_timer > 0:
            toast_timer -= 1
        else:
            toast = None
        clock.tick(60)


if __name__ == "__main__":
    from lerobot.common.robots.hope_jr import HopeJrHand, HopeJrHandConfig

    cfg = HopeJrHandConfig("/dev/tty.usbmodem58760431541")
    hand = HopeJrHand()
    main(hand.bus)
