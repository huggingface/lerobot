import numpy as np
import pytest

from examples.picklift_v3.operator_ui import OperatorUI, render_dashboard


def test_dashboard_renders_expected_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dashboard = render_dashboard(
        frame,
        status="RECORDING",
        elapsed_s=1.5,
        frames=30,
        target_frames=200,
        message="engineering smoke",
    )
    assert dashboard.shape == (760, 1280, 3)
    assert dashboard.dtype == np.uint8
    assert np.count_nonzero(dashboard) > 0


def test_dashboard_rejects_noncanonical_front():
    with pytest.raises(ValueError, match="expected RGB"):
        render_dashboard(np.zeros((720, 1280, 3), dtype=np.uint8), status="WAITING")


def test_show_acknowledges_enabled_click_once(monkeypatch):
    ui = OperatorUI(target_frames=20)
    ui._mouse_command = "start"
    acknowledgements = []
    monkeypatch.setattr(
        ui,
        "_acknowledge",
        lambda _frame, **kwargs: acknowledgements.append(kwargs),
    )
    monkeypatch.setattr("cv2.imshow", lambda *_args: None)
    monkeypatch.setattr("cv2.waitKey", lambda _delay: -1)

    command = ui.show(
        np.zeros((480, 640, 3), dtype=np.uint8),
        status="WAITING",
        buttons_enabled=(True, False, True),
    )

    assert command == "start"
    assert len(acknowledgements) == 1
    assert ui._mouse_command is None


def test_show_ignores_disabled_button_without_feedback(monkeypatch):
    ui = OperatorUI(target_frames=20)
    ui._mouse_command = "stop"
    acknowledgements = []
    monkeypatch.setattr(
        ui,
        "_acknowledge",
        lambda _frame, **kwargs: acknowledgements.append(kwargs),
    )
    monkeypatch.setattr("cv2.imshow", lambda *_args: None)
    monkeypatch.setattr("cv2.waitKey", lambda _delay: -1)

    command = ui.show(
        np.zeros((480, 640, 3), dtype=np.uint8),
        status="WAITING",
        buttons_enabled=(True, False, True),
    )

    assert command is None
    assert acknowledgements == []


def test_show_drops_repeat_click_during_transition_lock(monkeypatch):
    ui = OperatorUI(target_frames=20)
    ui._mouse_command = "start"
    ui._input_lock_until = float("inf")
    acknowledgements = []
    monkeypatch.setattr(
        ui,
        "_acknowledge",
        lambda _frame, **kwargs: acknowledgements.append(kwargs),
    )
    monkeypatch.setattr("cv2.imshow", lambda *_args: None)
    monkeypatch.setattr("cv2.waitKey", lambda _delay: -1)

    command = ui.show(
        np.zeros((480, 640, 3), dtype=np.uint8),
        status="REVIEW",
        button_labels=("CONFIRM", "BACK", "DISCARD"),
    )

    assert command is None
    assert acknowledgements == []


def test_success_selection_requires_manual_criteria_confirmation(monkeypatch):
    ui = OperatorUI(target_frames=20)
    commands = iter(("start", None, "start"))
    screens = []

    def fake_show(_frame, **kwargs):
        screens.append(kwargs)
        return next(commands)

    monkeypatch.setattr(ui, "show", fake_show)
    monkeypatch.setattr("examples.picklift_v3.operator_ui.time.sleep", lambda _seconds: None)

    result = ui.review_result(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result == "success"
    assert screens[0]["button_labels"] == ("SUCCESS", "FAILURE", "DISCARD")
    assert screens[1]["button_labels"] == ("CONFIRM", "BACK", "DISCARD")
    assert "lift >=5cm" in screens[1]["message"]
    assert "both fingers" in screens[1]["message"]
    assert "Held >=0.5s" in screens[1]["message"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("stop", "failure"),
        ("quit", "discard"),
    ],
)
def test_non_success_result_does_not_claim_automatic_detection(monkeypatch, command, expected):
    ui = OperatorUI(target_frames=20)
    screens = []

    def fake_show(_frame, **kwargs):
        screens.append(kwargs)
        return command

    monkeypatch.setattr(ui, "show", fake_show)

    assert ui.review_result(np.zeros((480, 640, 3), dtype=np.uint8)) == expected
    assert "manual visual" in screens[0]["message"]
