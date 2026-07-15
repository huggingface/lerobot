import sys
from types import SimpleNamespace

import numpy as np

from lerobot.utils.video_annotation import annotate_frame


def test_overlay_draws_each_label_once(monkeypatch):
    put_text_calls = []
    rectangle_calls = []

    def put_text(image, text, origin, font, scale, color, thickness, line_type):
        put_text_calls.append((text, color, thickness))
        return image

    def rectangle(image, start, end, color, thickness):
        rectangle_calls.append((start, end, color, thickness))
        return image

    def add_weighted(src1, alpha, src2, beta, gamma, *, dst):
        dst[:] = src1 * alpha + src2 * beta + gamma
        return dst

    fake_cv2 = SimpleNamespace(
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=16,
        getTextSize=lambda text, font, scale, thickness: ((len(text) * 7, 10), 0),
        putText=put_text,
        rectangle=rectangle,
        addWeighted=add_weighted,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    frame = np.full((120, 480, 3), 200, dtype=np.uint8)
    annotated = annotate_frame(
        frame,
        (("Task", "close the fridge"), ("Subtask", "reach for the handle"), ("Memory", None)),
    )

    assert [call[0] for call in put_text_calls] == [
        "Task: close the fridge",
        "Subtask: reach for the handle",
    ]
    assert all(color == (255, 255, 255) and thickness == 1 for _, color, thickness in put_text_calls)
    assert len(rectangle_calls) == 1
    assert not np.shares_memory(annotated, frame)
