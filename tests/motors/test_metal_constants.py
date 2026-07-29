from lerobot.motors.damiao.tables import MODEL_NAMES, MOTOR_LIMIT_PARAMS, MotorType


def test_metal_motor_types_registered():
    for name in ("metal_jlo", "metal_j2", "metal_jhi"):
        assert name in MODEL_NAMES.values()
    # (PMAX rad, VMAX rad/s, TMAX N*m) -- these are the MIT fixed-point encoding scales, so a
    # wrong entry silently mis-encodes every commanded position, velocity and torque.
    lim = {MODEL_NAMES[t]: MOTOR_LIMIT_PARAMS[t] for t in MotorType}
    assert lim["metal_jlo"] == (6.28, 10, 30)
    assert lim["metal_j2"] == (6.28, 10, 120)
    assert lim["metal_jhi"] == (6.28, 30, 20)
