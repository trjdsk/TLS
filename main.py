"""Touchless Lock System - Main Pipeline.

Complete palm detection and verification pipeline:
1. Camera stream with MediaPipe palm detection
2. LBP feature extraction and hand geometry analysis
3. Verification against registered users using feature similarity
4. ESP32 communication for unlock signals

Run:
    python main.py
Press 'q' to quit, 'r' to register new user.
"""
from __future__ import annotations

import os
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
import sys
import time

import cv2
import numpy as np

from camera import VideoCapture
from detector import PalmDetector, PalmDetection
from registration import PalmRegistrar
from verification import verify_palm_with_features

registrar = PalmRegistrar(use_geometry=True)


@dataclass
class AppConfig:
    source: Union[int, str] = 0
    width: Optional[int] = None
    height: Optional[int] = None
    max_fps: Optional[float] = 30.0
    display: bool = True
    camera_buffer_size: Optional[int] = None
    cv2_threads: Optional[int] = None
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    use_geometry: bool = True
    similarity_threshold: float = 0.92
    esp32_enabled: bool = False
    esp32_port: Optional[str] = None
    save_snaps: bool = False
    snaps_dir: Optional[str] = "snapshots"


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Touchless Lock System - Palm Detection & Verification")
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--log-level", default="DEBUG")
    parser.add_argument("--max-fps", type=float, default=30.0)
    parser.add_argument("--detection-confidence", type=float, default=0.7)
    parser.add_argument("--tracking-confidence", type=float, default=0.6)
    parser.add_argument("--use-geometry", action="store_true", default=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    parser.add_argument("--camera-buffer-size", type=int, default=None)
    parser.add_argument("--cv2-threads", type=int, default=None)
    parser.add_argument("--esp32-enabled", action="store_true")
    parser.add_argument("--esp32-port", type=str, default=None)
    parser.add_argument("--save-snaps", action="store_true")
    parser.add_argument("--snaps-dir", type=str, default="snapshots")
    try:
        bool_action = argparse.BooleanOptionalAction
        parser.add_argument("--display", action=bool_action, default=True)
    except Exception:
        parser.add_argument("--display", action="store_true", default=True)

    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    setup_logging(args.log_level)

    max_fps: Optional[float]
    if args.max_fps is None or args.max_fps <= 0:
        max_fps = None
    else:
        max_fps = float(args.max_fps)

    return AppConfig(
        source=source,
        width=args.width,
        height=args.height,
        max_fps=max_fps,
        display=getattr(args, "display", True),
        camera_buffer_size=args.camera_buffer_size,
        cv2_threads=args.cv2_threads,
        detection_confidence=float(args.detection_confidence),
        tracking_confidence=float(args.tracking_confidence),
        use_geometry=getattr(args, "use_geometry", True),
        similarity_threshold=float(args.similarity_threshold),
        esp32_enabled=getattr(args, "esp32_enabled", False),
        esp32_port=args.esp32_port,
        save_snaps=getattr(args, "save_snaps", False),
        snaps_dir=getattr(args, "snaps_dir", "snapshots"),
    )


def print_status(message: str) -> None:
    try:
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()
    except Exception:
        pass


def send_esp32_signal(unlock: bool, config: AppConfig, logger: logging.Logger) -> None:
    if not config.esp32_enabled:
        return
    try:
        signal = "UNLOCK" if unlock else "LOCK"
        logger.info(f"ESP32 Signal: {signal}")
        print_status(f"ESP32 Signal: {signal}")
    except Exception as exc:
        logger.error(f"Failed to send ESP32 signal: {exc}")


def main() -> None:
    config = parse_args()
    logger = logging.getLogger("app")
    logger.info("Starting Touchless Lock System | source=%s", str(config.source))

    try:
        cv2.setUseOptimized(True)
        if config.cv2_threads is not None and config.cv2_threads > 0:
            cv2.setNumThreads(int(config.cv2_threads))
    except Exception:
        pass

    detector = PalmDetector(
        max_num_hands=2,
        detection_confidence=config.detection_confidence,
        tracking_confidence=config.tracking_confidence
    )

    cap = VideoCapture(source=config.source, width=config.width, height=config.height, 
                      buffer_size=config.camera_buffer_size, detector=detector,
                      save_snaps=config.save_snaps, snaps_dir=config.snaps_dir)
    try:
        cap.open()
    except Exception as exc:
        logger.error("Could not open video source: %s", exc)
        return

    window_name = "Touchless Lock System"
    if config.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, 640, 480)
        except Exception:
            pass

    status_text = "Touchless Lock System Ready. Press 'r' to Register, 'q' to quit."

    registering: bool = False
    registration_detections: list[PalmDetection] = []
    registration_targets: int = 10
    registration_handedness: Optional[str] = None
    
    verification_detections: list[PalmDetection] = []
    verification_targets: int = 5
    verification_cooldown: Optional[float] = None

    frame_index: int = 0
    try:
        while True:
            frame_start = time.perf_counter()
            frame_index += 1
            
            now = time.time()
            if verification_cooldown is not None and now < verification_cooldown:
                remaining = verification_cooldown - now
                status_text = f"Cooldown: {remaining:.1f}s remaining"
                success, annotated, palm_detections = cap.get_palm_frame()
                if not success:
                    logger.warning("No frame received from source.")
                    break
            else:
                success, annotated, palm_detections = cap.get_palm_frame()
                if not success:
                    logger.warning("No frame received from source.")
                    break
                if palm_detections:
                    logger.info("Frame %d: %d palm detections", frame_index, len(palm_detections))
            
            if registering:
                if palm_detections:
                    palm_detection = palm_detections[0]
                    registration_detections.append(palm_detection)
                    status_text = f"Registration: {len(registration_detections)}/{registration_targets} detections"
                    
                    if len(registration_detections) >= registration_targets:
                        try:
                            user_name = input("\nRegistration complete! Enter your name: ").strip() or "Unknown"
                            success, user_id = registrar.register_user_with_features(
                                registration_detections,
                                handedness=registration_handedness or "Right",
                                name=user_name
                            )
                            if success:
                                status_text = f"Registration successful for {user_name}!"
                                print_status(f"Registration successful for {user_name} ({registration_handedness or 'Right'} hand)!")
                            else:
                                status_text = "Registration failed"
                                print_status("Registration failed")
                        except Exception as exc:
                            logger.error(f"Registration error: {exc}")
                            status_text = "Registration failed"
                            print_status("Registration failed")
                        registering = False
                        registration_detections.clear()
                        registration_handedness = None
            else:
                if palm_detections:
                    verification_detections.append(palm_detections[0])
                    status_text = f"Verification: {len(verification_detections)}/{verification_targets} detections"
                    
                    if len(verification_detections) >= verification_targets:
                        try:
                            is_match, matched_user_id, matched_name = verify_palm_with_features(
                                verification_detections,
                                handedness=None,
                                use_geometry=config.use_geometry,
                                similarity_threshold=config.similarity_threshold
                            )
                            if is_match:
                                display_name = matched_name if matched_name and matched_name != "Unknown" else f"User {matched_user_id}"
                                status_text = f"Access Granted: {display_name}"
                                print_status(f"Access Granted: {display_name}")
                                send_esp32_signal(True, config, logger)
                            else:
                                status_text = "Access Denied"
                                print_status("Access Denied")
                                send_esp32_signal(False, config, logger)
                        except Exception as exc:
                            logger.error(f"Verification error: {exc}")
                            status_text = "Verification failed"
                            print_status("Verification failed")
                        verification_detections.clear()
                        verification_cooldown = now + 2.0
                else:
                    status_text = "No valid palm detected. Press 'r' to Register, 'q' to quit."

            if config.display:
                try:
                    cv2.putText(annotated, status_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception:
                    pass
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0

            if key == ord("q"):
                logger.info("Quit requested by user.")
                break
            elif key == ord("r") and not registering:
                registering = True
                registration_detections.clear()
                registration_handedness = None
                status_text = "Registration started. Show your palm..."
                print_status(status_text)
            elif key == ord("c") and registering:
                registering = False
                registration_detections.clear()
                registration_handedness = None
                status_text = "Registration cancelled."
                print_status(status_text)

            if config.max_fps is not None and config.max_fps > 0:
                target_dt = 1.0 / max(1e-6, config.max_fps)
                elapsed = time.perf_counter() - frame_start
                remaining = target_dt - elapsed
                if remaining > 0:
                    try:
                        time.sleep(remaining)
                    except Exception:
                        pass

    finally:
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass
        try:
            if 'detector' in locals():
                detector.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        if config.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
