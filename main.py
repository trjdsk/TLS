"""Entry point for real-time palm/hand detection demo.

Run:
    python main.py
Press 'q' to quit.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

from camera import VideoCapture
from detector import PalmDetector


@dataclass
class AppConfig:
    backend: str = "mediapipe"
    source: Union[int, str] = 0
    width: Optional[int] = None
    height: Optional[int] = None
    immediate_forwarding: bool = True
    buffered_mode: bool = False
    palm_only: bool = False
    palm_region_only: bool = False


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Palm detection with pluggable backends")
    parser.add_argument("--backend", choices=["mediapipe", "edgeimpulse"], default="mediapipe", help="Detection backend")
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0) or stream URL")
    parser.add_argument("--width", type=int, default=None, help="Desired frame width")
    parser.add_argument("--height", type=int, default=None, help="Desired frame height")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--palm-only", action="store_true", help="Filter to open palm detections only (heuristic)")
    parser.add_argument("--palm-region-only", action="store_true", help="Draw and return only the palm region (no fingers)")
    try:
        bool_action = argparse.BooleanOptionalAction  # Python 3.9+
        parser.add_argument("--immediate-forwarding", action=bool_action, default=True, help="ESP32 streams directly for detection (placeholder)")
        parser.add_argument("--buffered-mode", action=bool_action, default=False, help="ESP32 buffered SD card mode (placeholder)")
    except Exception:
        parser.add_argument("--immediate-forwarding", action="store_true", default=True, help="ESP32 streams directly for detection (placeholder)")
        parser.add_argument("--buffered-mode", action="store_true", default=False, help="ESP32 buffered SD card mode (placeholder)")
    args = parser.parse_args()

    # Convert source to int if numeric
    source: Union[int, str]
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    setup_logging(args.log_level)

    return AppConfig(
        backend=args.backend,
        source=source,
        width=args.width,
        height=args.height,
        immediate_forwarding=getattr(args, "immediate_forwarding", True),
        buffered_mode=getattr(args, "buffered_mode", False),
        palm_only=bool(args.palm_only),
        palm_region_only=bool(args.palm_region_only),
    )


def main() -> None:
    config = parse_args()
    logger = logging.getLogger("app")
    logger.info("Starting palm detection | backend=%s | source=%s", config.backend, str(config.source))

    if config.buffered_mode:
        logger.info("Buffered mode requested. Placeholder only; not implemented.")
    if config.immediate_forwarding:
        logger.info("Immediate forwarding enabled (placeholder). Using local capture.")

    cap = VideoCapture(source=config.source, width=config.width, height=config.height)
    try:
        cap.open()
    except Exception as exc:
        logger.error("Could not open video source: %s", exc)
        return

    detector = PalmDetector(backend=config.backend, palm_only=config.palm_only, palm_region_only=config.palm_region_only)

    window_name = "Palm Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("No frame received from source.")
                break

            annotated, detections = detector.detect(frame)

            if len(detections) > 0:
                logger.info("Palm detected (%d hands)", len(detections))
            else:
                logger.debug("No palm detected")

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested by user.")
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
