#!/usr/bin/env python

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2

from lerobot.configs import parser
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class MakeTagConfig:
    dictionary: str = "apriltag_36h11"
    tag_id: int = 0
    # Pixel size of the generated marker (square). Bigger prints detect easier.
    size_px: int = 800
    out: str = "tag.png"


@parser.wrap()
def make_tag(cfg: MakeTagConfig):
    init_logging()

    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "OpenCV ArUco module not found. Install opencv-contrib-python to generate tags."
        )

    dict_map = {
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "aruco_4x4_50": cv2.aruco.DICT_4X4_50,
        "aruco_4x4_100": cv2.aruco.DICT_4X4_100,
    }
    if cfg.dictionary not in dict_map:
        raise ValueError(f"Unknown dictionary '{cfg.dictionary}'. Options: {sorted(dict_map.keys())}")

    d = cv2.aruco.getPredefinedDictionary(dict_map[cfg.dictionary])

    # Generate a black/white marker.
    img = cv2.aruco.generateImageMarker(d, int(cfg.tag_id), int(cfg.size_px))
    ok = cv2.imwrite(cfg.out, img)
    if not ok:
        raise RuntimeError(f"Failed to write output image to {cfg.out}")

    logger.info("Wrote tag: dict=%s id=%d size_px=%d -> %s", cfg.dictionary, cfg.tag_id, cfg.size_px, cfg.out)
    logger.info("Print tip: avoid scaling artifacts; print at high quality with a clear white border.")


def main():
    make_tag()


if __name__ == "__main__":
    main()

