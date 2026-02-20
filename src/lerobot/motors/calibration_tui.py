"""Curses-based per-joint calibration TUI shared by leader and follower arms."""

import curses

from .feetech import FeetechMotorsBus

HALF = 2047
MAX_POS = 4095
RESOLUTION = MAX_POS + 1  # 4096

# Color pair IDs
C_AMBER = 1
C_RED = 2
C_GREEN = 3


def run_calibration_tui(
    bus: FeetechMotorsBus,
    title: str = "CALIBRATION",
    full_turn_motor: str = "wrist_roll",
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Per-joint endpoint calibration TUI.

    For each joint, the user moves to one physical end, presses ENTER,
    then moves to the other end and presses ENTER.  For the full-turn motor
    (wrist_roll), the user picks a comfortable center instead.

    Continuously tracks observed min/max during each joint's flow for
    ground-truth range capture.

    Returns:
        (homing_offsets, range_mins, range_maxes) dicts keyed by motor name.
    """
    motor_names = list(bus.motors.keys())
    endpoints: dict[str, tuple[int, int]] = {}
    ft_center: dict[str, int] = {}

    def run(stdscr: curses.window) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(C_AMBER, curses.COLOR_YELLOW, -1)
        curses.init_pair(C_RED, curses.COLOR_RED, -1)
        curses.init_pair(C_GREEN, curses.COLOR_GREEN, -1)

        def draw_gauge(row: int, x: int, w: int, pos: int,
                       end1: int | None = None) -> None:
            h_scr, w_scr = stdscr.getmaxyx()
            if row >= h_scr:
                return
            bar = [0] * w
            pos_i = max(0, min(w - 1, round(pos / MAX_POS * (w - 1))))

            if end1 is not None:
                e1_i = max(0, min(w - 1, round(end1 / MAX_POS * (w - 1))))
                bar[e1_i] = 1

            for d in (-1, 0, 1):
                i = pos_i + d
                if 0 <= i < w:
                    bar[i] = 2

            for i, g in enumerate(bar):
                if x + i >= w_scr:
                    break
                try:
                    if g == 2:
                        stdscr.addstr(row, x + i, "█",
                                      curses.color_pair(C_AMBER) | curses.A_BOLD)
                    elif g == 1:
                        stdscr.addstr(row, x + i, "▎",
                                      curses.color_pair(C_RED) | curses.A_BOLD)
                    else:
                        stdscr.addstr(row, x + i, "░", curses.A_DIM)
                except curses.error:
                    pass

        def draw_range_bar(row: int, x: int, w: int,
                           lo: int, hi: int) -> None:
            h_scr, w_scr = stdscr.getmaxyx()
            if row >= h_scr:
                return
            lo_i = round(lo / MAX_POS * (w - 1))
            hi_i = round(hi / MAX_POS * (w - 1))
            for i in range(w):
                if x + i >= w_scr:
                    break
                try:
                    if lo_i <= i <= hi_i:
                        stdscr.addstr(row, x + i, "━",
                                      curses.color_pair(C_AMBER))
                    else:
                        stdscr.addstr(row, x + i, "─", curses.A_DIM)
                except curses.error:
                    pass

        def draw_screen(motor: str, instruction: str,
                        step: int, total: int,
                        pos: int, end1: int | None) -> None:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            max_row = h - 1  # last safe row for addstr
            bar_w = max(30, min(60, w - 16))
            bar_x = 8
            summary_bar_w = max(12, min(24, w - 50))

            def safe_addstr(r: int, c: int, text: str, attr: int = 0) -> None:
                if r > max_row:
                    return
                # Truncate text if it would exceed the width
                max_len = max(0, w - c)
                if len(text) > max_len:
                    text = text[:max_len]
                if text:
                    try:
                        stdscr.addstr(r, c, text, attr)
                    except curses.error:
                        pass

            row = 1

            # Title bar
            safe_addstr(row, 3, title,
                        curses.color_pair(C_AMBER) | curses.A_BOLD)
            step_str = f"{step}/{total}"
            safe_addstr(row, w - len(step_str) - 3, step_str, curses.A_DIM)
            row += 1
            safe_addstr(row, 3, "━" * (w - 6),
                        curses.color_pair(C_RED))
            row += 3

            # Active joint
            safe_addstr(row, 5, "▸",
                        curses.color_pair(C_RED) | curses.A_BOLD)
            safe_addstr(row, 7, motor,
                        curses.color_pair(C_AMBER) | curses.A_BOLD)
            row += 2
            safe_addstr(row, 7, instruction, curses.A_DIM)
            row += 3

            # Gauge
            if row <= max_row:
                draw_gauge(row, bar_x, bar_w, pos, end1)
            row += 1

            # Scale
            safe_addstr(row, bar_x, "0", curses.A_DIM)
            safe_addstr(row, bar_x + bar_w - 4, "4095", curses.A_DIM)
            row += 1

            # Position value under indicator
            pos_str = str(pos)
            pos_i = max(0, min(bar_w - 1,
                               round(pos / MAX_POS * (bar_w - 1))))
            vx = bar_x + pos_i - len(pos_str) // 2
            vx = max(bar_x, min(bar_x + bar_w - len(pos_str), vx))
            safe_addstr(row, vx, pos_str,
                        curses.color_pair(C_AMBER) | curses.A_BOLD)

            # End1 label if present
            if end1 is not None:
                e1_i = max(0, min(bar_w - 1,
                                  round(end1 / MAX_POS * (bar_w - 1))))
                e1_str = str(end1)
                e1x = bar_x + e1_i - len(e1_str) // 2
                e1x = max(bar_x, min(bar_x + bar_w - len(e1_str), e1x))
                if abs(e1x - vx) > max(len(pos_str), len(e1_str)) + 1:
                    safe_addstr(row, e1x, e1_str,
                                curses.color_pair(C_RED))

            row += 4

            # Separator
            safe_addstr(row, 3, "━" * (w - 6),
                        curses.color_pair(C_RED))
            row += 2

            # Joint list
            for name in motor_names:
                if row > max_row:
                    break
                if name in endpoints:
                    lo, hi = endpoints[name]
                    safe_addstr(row, 5, "✓",
                                curses.color_pair(C_GREEN) | curses.A_BOLD)
                    safe_addstr(row, 8, f"{name:<16}",
                                curses.color_pair(C_GREEN))
                    bx = 25
                    if row <= max_row:
                        draw_range_bar(row, bx, summary_bar_w, lo, hi)
                    safe_addstr(row, bx + summary_bar_w + 2,
                                f"{lo} — {hi}", curses.A_DIM)
                elif name == motor:
                    safe_addstr(row, 5, "▸",
                                curses.color_pair(C_RED) | curses.A_BOLD)
                    safe_addstr(row, 8, f"{name:<16}",
                                curses.color_pair(C_AMBER))
                elif name == full_turn_motor and name not in endpoints:
                    safe_addstr(row, 5, "○", curses.A_DIM)
                    safe_addstr(row, 8, f"{name:<16}", curses.A_DIM)
                    safe_addstr(row, 25, "(360°)", curses.A_DIM)
                else:
                    safe_addstr(row, 5, "○", curses.A_DIM)
                    safe_addstr(row, 8, f"{name:<16}", curses.A_DIM)
                row += 1

            row += 2
            safe_addstr(row, 5, "ENTER to confirm",
                        curses.A_DIM)

            stdscr.refresh()

        # Per-motor tracking of observed min/max across both phases
        observed_min: dict[str, int] = {}
        observed_max: dict[str, int] = {}

        def wait_for_enter(motor: str, instruction: str,
                           step: int, total: int,
                           end1: int | None = None) -> int:
            while True:
                positions = bus.sync_read(
                    "Present_Position", normalize=False)
                pos = positions[motor]
                observed_min[motor] = min(observed_min.get(motor, pos), pos)
                observed_max[motor] = max(observed_max.get(motor, pos), pos)
                draw_screen(motor, instruction, step, total, pos, end1)
                key = stdscr.getch()
                if key in (curses.KEY_ENTER, 10, 13):
                    return pos

        total = sum(1 if m == full_turn_motor else 2 for m in motor_names)
        step = 0

        for motor in motor_names:
            observed_min.pop(motor, None)
            observed_max.pop(motor, None)

            if motor == full_turn_motor:
                step += 1
                mid = wait_for_enter(
                    motor,
                    "Move to a comfortable center position, then press ENTER",
                    step, total)
                ft_center[motor] = mid
                endpoints[motor] = (0, MAX_POS)
            else:
                step += 1
                wait_for_enter(
                    motor,
                    "Move to one physical end, then press ENTER",
                    step, total)
                end1_display = observed_min[motor]
                step += 1
                wait_for_enter(
                    motor,
                    "Move to the OTHER physical end, then press ENTER",
                    step, total, end1_display)
                endpoints[motor] = (observed_min[motor], observed_max[motor])

    curses.wrapper(run)

    # Compute homing offsets and homed ranges
    homing_offsets: dict[str, int] = {}
    range_mins: dict[str, int] = {}
    range_maxes: dict[str, int] = {}
    for motor in motor_names:
        lo_raw, hi_raw = endpoints[motor]
        if motor == full_turn_motor:
            mid_single = ft_center[motor] % RESOLUTION
            offset = mid_single - HALF
            offset = max(-HALF, min(HALF, offset))
            homing_offsets[motor] = offset
            range_mins[motor] = 0
            range_maxes[motor] = MAX_POS
        else:
            physical_range = abs(hi_raw - lo_raw)
            mid_raw = (lo_raw + hi_raw) // 2
            mid_single = mid_raw % RESOLUTION
            offset = mid_single - HALF
            offset = max(-HALF, min(HALF, offset))
            homing_offsets[motor] = offset
            half_range = physical_range // 2
            range_mins[motor] = HALF - half_range
            range_maxes[motor] = HALF + half_range

    return homing_offsets, range_mins, range_maxes
