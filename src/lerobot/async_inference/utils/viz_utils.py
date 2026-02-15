import logging


# Track which (schedule, d, overlap_end) combos have been logged to avoid spam
_prefix_weights_logged: set[tuple[str, int, int]] = set()


def compute_prefix_weights_for_viz(d: int, overlap_end: int, H: int, schedule: str = "linear") -> list[float]:
    """Compute prefix weights for RTC visualization.

    Args:
        d: Inference delay (hard mask region ends at d).
        overlap_end: Where soft masking ends (H - d with s=d).
        H: Total chunk size.
        schedule: Weight schedule ("linear" or "exp").

    Returns:
        List of H floats, each in [0, 1]:
        - [0, d): weight = 1.0 (hard mask)
        - [d, overlap_end): weight decays 1->0 (soft mask)
        - [overlap_end, H): weight = 0.0 (fresh)
    """
    import math

    weights = []
    for i in range(H):
        if i < d:
            # Hard mask region
            weights.append(1.0)
        elif i < overlap_end:
            # Soft masking region - linear decay from 1 to 0
            if overlap_end > d:
                t = (i - d) / (overlap_end - d)  # t goes from 0 to 1
                w = 1.0 - t  # Linear decay
                if schedule.lower() == "exp":
                    # Exponential decay (steeper at start)
                    w = w * (math.expm1(w) / (math.e - 1)) if w > 0 else 0.0
                weights.append(w)
            else:
                weights.append(0.0)
        else:
            # Fresh region
            weights.append(0.0)

    # Log weight samples once per unique (schedule, d, overlap_end) to verify formula
    _log_key = (schedule.lower(), d, overlap_end)
    if _log_key not in _prefix_weights_logged and H > 0:
        _prefix_weights_logged.add(_log_key)
        logger = logging.getLogger("policy_server_drtc")
        sample_indices = [d, (d + overlap_end) // 2, overlap_end - 1]
        samples = [(i, weights[i]) for i in sample_indices if 0 <= i < len(weights)]
        logger.info(
            "RTC prefix weights (%s): d=%d, overlap_end=%d, H=%d, samples=%s",
            schedule, d, overlap_end, H,
            [(f"w[{i}]", f"{w:.3f}") for i, w in samples],
        )

    return weights
