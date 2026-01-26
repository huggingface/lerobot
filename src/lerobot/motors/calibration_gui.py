# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from dataclasses import dataclass

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from .motors_bus import MotorCalibration, MotorsBus

BAR_LEN, BAR_THICKNESS = 450, 8
HANDLE_R = 10
BRACKET_W, BRACKET_H = 6, 14
TRI_W, TRI_H = 12, 14

BTN_W, BTN_H = 60, 22
SAVE_W, SAVE_H = 80, 28
LOAD_W = 80
DD_W, DD_H = 160, 28

TOP_GAP = 50
PADDING_Y, TOP_OFFSET = 70, 60
FONT_SIZE, FPS = 20, 60

BG_COLOR = (30, 30, 30)
BAR_RED, BAR_GREEN = (200, 60, 60), (60, 200, 60)
HANDLE_COLOR, TEXT_COLOR = (240, 240, 240), (250, 250, 250)
TICK_COLOR = (250, 220, 40)
BTN_COLOR, BTN_COLOR_HL = (80, 80, 80), (110, 110, 110)
DD_COLOR, DD_COLOR_HL = (70, 70, 70), (100, 100, 100)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class RangeValues:
    min_v: int
    pos_v: int
    max_v: int


class RangeSlider:
    """One motor = one slider row"""

    def __init__(self, motor, idx, res, calibration, present, label_pad, base_y):
        import pygame

        self.motor = motor
        self.res = res
        self.x0 = 40 + label_pad
        self.x1 = self.x0 + BAR_LEN
        self.y = base_y + idx * PADDING_Y

        self.min_v = calibration.range_min
        self.max_v = calibration.range_max
        self.pos_v = max(self.min_v, min(present, self.max_v))

        self.min_x = self._pos_from_val(self.min_v)
        self.max_x = self._pos_from_val(self.max_v)
        self.pos_x = self._pos_from_val(self.pos_v)

        self.min_btn = pygame.Rect(self.x0 - BTN_W - 6, self.y - BTN_H // 2, BTN_W, BTN_H)
        self.max_btn = pygame.Rect(self.x1 + 6, self.y - BTN_H // 2, BTN_W, BTN_H)

        self.drag_min = self.drag_max = self.drag_pos = False
        self.tick_val = present
        self.font = pygame.font.Font(None, FONT_SIZE)

    def _val_from_pos(self, x):
        return round((x - self.x0) / BAR_LEN * self.res)

    def _pos_from_val(self, v):
        return self.x0 + (v / self.res) * BAR_LEN

    def set_tick(self, v):
        self.tick_val = max(0, min(v, self.res))

    def _triangle_hit(self, pos):
        import pygame

        tri_top = self.y - BAR_THICKNESS // 2 - 2
        return pygame.Rect(self.pos_x - TRI_W // 2, tri_top - TRI_H, TRI_W, TRI_H).collidepoint(pos)

    def handle_event(self, e):
        import pygame

        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.min_btn.collidepoint(e.pos):
                self.min_x, self.min_v = self.pos_x, self.pos_v
                return
            if self.max_btn.collidepoint(e.pos):
                self.max_x, self.max_v = self.pos_x, self.pos_v
                return
            if dist(e.pos, (self.min_x, self.y)) <= HANDLE_R:
                self.drag_min = True
            elif dist(e.pos, (self.max_x, self.y)) <= HANDLE_R:
                self.drag_max = True
            elif self._triangle_hit(e.pos):
                self.drag_pos = True

        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.drag_min = self.drag_max = self.drag_pos = False

        elif e.type == pygame.MOUSEMOTION:
            x = e.pos[0]
            if self.drag_min:
                self.min_x = max(self.x0, min(x, self.pos_x))
            elif self.drag_max:
                self.max_x = min(self.x1, max(x, self.pos_x))
            elif self.drag_pos:
                self.pos_x = max(self.min_x, min(x, self.max_x))

            self.min_v = self._val_from_pos(self.min_x)
            self.max_v = self._val_from_pos(self.max_x)
            self.pos_v = self._val_from_pos(self.pos_x)

    def _draw_button(self, surf, rect, text):
        import pygame

        clr = BTN_COLOR_HL if rect.collidepoint(pygame.mouse.get_pos()) else BTN_COLOR
        pygame.draw.rect(surf, clr, rect, border_radius=4)
        t = self.font.render(text, True, TEXT_COLOR)
        surf.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))

    def draw(self, surf):
        import pygame

        # motor name above set-min button (right-aligned)
        name_surf = self.font.render(self.motor, True, TEXT_COLOR)
        surf.blit(
            name_surf,
            (self.min_btn.right - name_surf.get_width(), self.min_btn.y - name_surf.get_height() - 4),
        )

        # bar + active section
        pygame.draw.rect(surf, BAR_RED, (self.x0, self.y - BAR_THICKNESS // 2, BAR_LEN, BAR_THICKNESS))
        pygame.draw.rect(
            surf, BAR_GREEN, (self.min_x, self.y - BAR_THICKNESS // 2, self.max_x - self.min_x, BAR_THICKNESS)
        )

        # tick
        tick_x = self._pos_from_val(self.tick_val)
        pygame.draw.line(
            surf,
            TICK_COLOR,
            (tick_x, self.y - BAR_THICKNESS // 2 - 4),
            (tick_x, self.y + BAR_THICKNESS // 2 + 4),
            2,
        )

        # brackets
        for x, sign in ((self.min_x, +1), (self.max_x, -1)):
            pygame.draw.line(
                surf, HANDLE_COLOR, (x, self.y - BRACKET_H // 2), (x, self.y + BRACKET_H // 2), 2
            )
            pygame.draw.line(
                surf,
                HANDLE_COLOR,
                (x, self.y - BRACKET_H // 2),
                (x + sign * BRACKET_W, self.y - BRACKET_H // 2),
                2,
            )
            pygame.draw.line(
                surf,
                HANDLE_COLOR,
                (x, self.y + BRACKET_H // 2),
                (x + sign * BRACKET_W, self.y + BRACKET_H // 2),
                2,
            )

        # triangle ▼
        tri_top = self.y - BAR_THICKNESS // 2 - 2
        pygame.draw.polygon(
            surf,
            HANDLE_COLOR,
            [
                (self.pos_x, tri_top),
                (self.pos_x - TRI_W // 2, tri_top - TRI_H),
                (self.pos_x + TRI_W // 2, tri_top - TRI_H),
            ],
        )

        # numeric labels
        fh = self.font.get_height()
        pos_y = tri_top - TRI_H - 4 - fh
        txts = [
            (self.min_v, self.min_x, self.y - BRACKET_H // 2 - 4 - fh),
            (self.max_v, self.max_x, self.y - BRACKET_H // 2 - 4 - fh),
            (self.pos_v, self.pos_x, pos_y),
        ]
        for v, x, y in txts:
            s = self.font.render(str(v), True, TEXT_COLOR)
            surf.blit(s, (x - s.get_width() // 2, y))

        # buttons
        self._draw_button(surf, self.min_btn, "set min")
        self._draw_button(surf, self.max_btn, "set max")

    # external
    def values(self) -> RangeValues:
        return RangeValues(self.min_v, self.pos_v, self.max_v)


class RangeFinderGUI:
    def __init__(self, bus: MotorsBus, groups: dict[str, list[str]] | None = None):
        import pygame

        self.bus = bus
        self.groups = groups if groups is not None else {"all": list(bus.motors)}
        self.group_names = list(groups)
        self.current_group = self.group_names[0]

        if not bus.is_connected:
            bus.connect()

        self.calibration = bus.read_calibration()
        self.res_table = bus.model_resolution_table
        self.present_cache = {
            m: bus.read("Present_Position", m, normalize=False) for motors in groups.values() for m in motors
        }

        pygame.init()
        self.font = pygame.font.Font(None, FONT_SIZE)

        label_pad = max(self.font.size(m)[0] for ms in groups.values() for m in ms)
        self.label_pad = label_pad
        width = 40 + label_pad + BAR_LEN + 6 + BTN_W + 10 + SAVE_W + 10
        self.controls_bottom = 10 + SAVE_H
        self.base_y = self.controls_bottom + TOP_GAP
        height = self.base_y + PADDING_Y * len(groups[self.current_group]) + 40

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Motors range finder")

        # ui rects
        self.save_btn = pygame.Rect(width - SAVE_W - 10, 10, SAVE_W, SAVE_H)
        self.load_btn = pygame.Rect(self.save_btn.left - LOAD_W - 10, 10, LOAD_W, SAVE_H)
        self.dd_btn = pygame.Rect(width // 2 - DD_W // 2, 10, DD_W, DD_H)
        self.dd_open = False  # dropdown expanded?

        self.clock = pygame.time.Clock()
        self._build_sliders()
        self._adjust_height()

    def _adjust_height(self):
        import pygame

        motors = self.groups[self.current_group]
        new_h = self.base_y + PADDING_Y * len(motors) + 40
        if new_h != self.screen.get_height():
            w = self.screen.get_width()
            self.screen = pygame.display.set_mode((w, new_h))

    def _build_sliders(self):
        self.sliders: list[RangeSlider] = []
        motors = self.groups[self.current_group]
        for i, m in enumerate(motors):
            self.sliders.append(
                RangeSlider(
                    motor=m,
                    idx=i,
                    res=self.res_table[self.bus.motors[m].model] - 1,
                    calibration=self.calibration[m],
                    present=self.present_cache[m],
                    label_pad=self.label_pad,
                    base_y=self.base_y,
                )
            )

    def _draw_dropdown(self):
        import pygame

        # collapsed box
        hover = self.dd_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, DD_COLOR_HL if hover else DD_COLOR, self.dd_btn, border_radius=6)

        txt = self.font.render(self.current_group, True, TEXT_COLOR)
        self.screen.blit(
            txt, (self.dd_btn.centerx - txt.get_width() // 2, self.dd_btn.centery - txt.get_height() // 2)
        )

        tri_w, tri_h = 12, 6
        cx = self.dd_btn.right - 14
        cy = self.dd_btn.centery + 1
        pygame.draw.polygon(
            self.screen,
            TEXT_COLOR,
            [(cx - tri_w // 2, cy - tri_h // 2), (cx + tri_w // 2, cy - tri_h // 2), (cx, cy + tri_h // 2)],
        )

        if not self.dd_open:
            return

        # expanded list
        for i, name in enumerate(self.group_names):
            item_rect = pygame.Rect(self.dd_btn.left, self.dd_btn.bottom + i * DD_H, DD_W, DD_H)
            clr = DD_COLOR_HL if item_rect.collidepoint(pygame.mouse.get_pos()) else DD_COLOR
            pygame.draw.rect(self.screen, clr, item_rect)
            t = self.font.render(name, True, TEXT_COLOR)
            self.screen.blit(
                t, (item_rect.centerx - t.get_width() // 2, item_rect.centery - t.get_height() // 2)
            )

    def _handle_dropdown_event(self, e):
        import pygame

        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.dd_btn.collidepoint(e.pos):
                self.dd_open = not self.dd_open
                return True
            if self.dd_open:
                for i, name in enumerate(self.group_names):
                    item_rect = pygame.Rect(self.dd_btn.left, self.dd_btn.bottom + i * DD_H, DD_W, DD_H)
                    if item_rect.collidepoint(e.pos):
                        if name != self.current_group:
                            self.current_group = name
                            self._build_sliders()
                            self._adjust_height()
                        self.dd_open = False
                        return True
                self.dd_open = False
        return False

    def _save_current(self):
        for s in self.sliders:
            self.calibration[s.motor].range_min = s.min_v
            self.calibration[s.motor].range_max = s.max_v

        with self.bus.torque_disabled():
            self.bus.write_calibration(self.calibration)

    def _load_current(self):
        self.calibration = self.bus.read_calibration()
        for s in self.sliders:
            s.min_v = self.calibration[s.motor].range_min
            s.max_v = self.calibration[s.motor].range_max
            s.min_x = s._pos_from_val(s.min_v)
            s.max_x = s._pos_from_val(s.max_v)

    def run(self) -> dict[str, MotorCalibration]:
        import pygame

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return self.calibration

                if self._handle_dropdown_event(e):
                    continue

                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    if self.save_btn.collidepoint(e.pos):
                        self._save_current()
                    elif self.load_btn.collidepoint(e.pos):
                        self._load_current()

                for s in self.sliders:
                    s.handle_event(e)

            # live goal write while dragging
            for s in self.sliders:
                if s.drag_pos:
                    self.bus.write("Goal_Position", s.motor, s.pos_v, normalize=False)

            # tick update
            for s in self.sliders:
                pos = self.bus.read("Present_Position", s.motor, normalize=False)
                s.set_tick(pos)
                self.present_cache[s.motor] = pos

            # ─ drawing
            self.screen.fill(BG_COLOR)
            for s in self.sliders:
                s.draw(self.screen)

            self._draw_dropdown()

            # load / save buttons
            for rect, text in ((self.load_btn, "LOAD"), (self.save_btn, "SAVE")):
                clr = BTN_COLOR_HL if rect.collidepoint(pygame.mouse.get_pos()) else BTN_COLOR
                pygame.draw.rect(self.screen, clr, rect, border_radius=6)
                t = self.font.render(text, True, TEXT_COLOR)
                self.screen.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))

            pygame.display.flip()
            self.clock.tick(FPS)
