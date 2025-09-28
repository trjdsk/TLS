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
from typing import Optional, Union, Any, Dict, List
import sys
import time
import math

import cv2
import numpy as np

from utils import config

from camera import VideoCapture
from detector import PalmDetector, DetectorConfig, PalmDetection
from registration import PalmRegistrar, RegistrarConfig
from verification import verify_palm_with_features
from utils.palm import is_palm_facing_camera
from utils import config

# Attempt to import a shared palm utility; if not available, use fallback
try:
    from utils.palm_utils import is_palm_facing_camera  # type: ignore
except Exception:
    is_palm_facing_camera = None  # will use fallback implementation below


@dataclass
class AppConfig:
    source: Union[int, str] = config.DEFAULT_SOURCE
    width: Optional[int] = config.DEFAULT_WIDTH
    height: Optional[int] = config.DEFAULT_HEIGHT
    max_fps: Optional[float] = config.DEFAULT_MAX_FPS
    display: bool = config.DISPLAY_DEFAULT
    camera_buffer_size: Optional[int] = config.DEFAULT_CAMERA_BUFFER_SIZE
    cv2_threads: Optional[int] = config.DEFAULT_CV2_THREADS
    detection_confidence: float = config.DETECTION_CONFIDENCE
    tracking_confidence: float = config.TRACKING_CONFIDENCE
    use_geometry: bool = config.DEFAULT_USE_GEOMETRY
    similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD
    esp32_enabled: bool = config.ESP32_ENABLED_DEFAULT
    esp32_port: Optional[str] = config.ESP32_PORT_DEFAULT
    save_snaps: bool = config.SAVE_SNAPS_DEFAULT
    snaps_dir: Optional[str] = str(config.SNAPS_DIR)


def setup_logging(level: str = None) -> None:
    level = level or config.LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Touchless Lock System - Palm Detection & Verification")
    parser.add_argument("--source", default=str(config.DEFAULT_SOURCE))
    parser.add_argument("--width", type=int, default=config.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=config.DEFAULT_HEIGHT)
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    parser.add_argument("--max-fps", type=float, default=config.DEFAULT_MAX_FPS)
    parser.add_argument("--detection-confidence", type=float, default=config.DETECTION_CONFIDENCE)
    parser.add_argument("--tracking-confidence", type=float, default=config.TRACKING_CONFIDENCE)
    parser.add_argument("--use-geometry", action="store_true", default=config.DEFAULT_USE_GEOMETRY)
    parser.add_argument("--similarity-threshold", type=float, default=config.DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--camera-buffer-size", type=int, default=config.DEFAULT_CAMERA_BUFFER_SIZE)
    parser.add_argument("--cv2-threads", type=int, default=config.DEFAULT_CV2_THREADS)
    parser.add_argument("--esp32-enabled", action="store_true", default=config.ESP32_ENABLED_DEFAULT)
    parser.add_argument("--esp32-port", type=str, default=config.ESP32_PORT_DEFAULT)
    parser.add_argument("--save-snaps", action="store_true", default=config.SAVE_SNAPS_DEFAULT)
    parser.add_argument("--snaps-dir", type=str, default=str(config.SNAPS_DIR))
    try:
        bool_action = argparse.BooleanOptionalAction
        parser.add_argument("--display", action=bool_action, default=config.DISPLAY_DEFAULT)
    except Exception:
        parser.add_argument("--display", action="store_true", default=config.DISPLAY_DEFAULT)

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
        display=getattr(args, "display", config.DISPLAY_DEFAULT),
        camera_buffer_size=args.camera_buffer_size,
        cv2_threads=args.cv2_threads,
        detection_confidence=float(args.detection_confidence),
        tracking_confidence=float(args.tracking_confidence),
        use_geometry=getattr(args, "use_geometry", config.DEFAULT_USE_GEOMETRY),
        similarity_threshold=float(args.similarity_threshold),
        esp32_enabled=getattr(args, "esp32_enabled", config.ESP32_ENABLED_DEFAULT),
        esp32_port=args.esp32_port,
        save_snaps=getattr(args, "save_snaps", config.SAVE_SNAPS_DEFAULT),
        snaps_dir=getattr(args, "snaps_dir", str(config.SNAPS_DIR)),
    )


def print_status(message: str) -> None:
    try:
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()
    except Exception:
        pass


def send_esp32_signal(unlock: bool, cfg: AppConfig, logger: logging.Logger) -> None:
    if not cfg.esp32_enabled:
        return
    try:
        signal = "UNLOCK" if unlock else "LOCK"
        logger.info(f"ESP32 Signal: {signal} (port={cfg.esp32_port})")
        print_status(f"ESP32 Signal: {signal}")
        # Here you would open a serial port and send the signal.
        # Keep hardware-specific code isolated so it can be mocked during tests.
    except Exception as exc:
        logger.error(f"Failed to send ESP32 signal: {exc}")


# Improved fallback palm-facing detection
def _fallback_is_palm_facing_camera(landmarks: List[Any], handedness: str = "Right") -> bool:
    try:
        def coord(lm, idx):
            lm_obj = lm[idx]
            if hasattr(lm_obj, "x"):
                return np.array([lm_obj.x, lm_obj.y, lm_obj.z], dtype=float)
            else:
                return np.array(lm_obj, dtype=float)

        a = coord(landmarks, 0)  # wrist
        b = coord(landmarks, 5)  # index mcp
        c = coord(landmarks, 9)  # middle mcp
        d = coord(landmarks, 17)  # pinky mcp

        v1 = b - a
        v2 = d - a
        normal = np.cross(v1, v2)
        forward = c - a
        cos_angle = np.dot(normal, forward) / ((np.linalg.norm(normal) * np.linalg.norm(forward)) + 1e-8)

        # Debug logging
        logging.debug(f"Palm facing cos_angle: {cos_angle:.3f}")

        # More permissive threshold
        return cos_angle > -0.3
    except Exception:
        return True  # default to True to not block verification


def main() -> None:
    app_cfg = parse_args()
    logger = logging.getLogger("app")
    logger.info("Starting Touchless Lock System | source=%s", str(app_cfg.source))

    # Setup cv2 threading/optimizations
    try:
        cv2.setUseOptimized(True)
        if app_cfg.cv2_threads is not None and app_cfg.cv2_threads > 0:
            cv2.setNumThreads(int(app_cfg.cv2_threads))
    except Exception:
        logger.debug("Failed to configure OpenCV threading/optimizations", exc_info=True)

    # instantiate detector and registrar here (use config values)
    detector = None
    cap = None
    registrar = None
    try:
        detector_cfg = DetectorConfig(
            min_detection_confidence=app_cfg.detection_confidence,
            min_tracking_confidence=app_cfg.tracking_confidence,
        )
        detector = PalmDetector(config=detector_cfg, max_num_hands=config.MAX_NUM_HANDS)

        registrar_config = RegistrarConfig(use_geometry=app_cfg.use_geometry)
        registrar = PalmRegistrar(config=registrar_config)

        cap = VideoCapture(
            source=app_cfg.source,
            width=app_cfg.width,
            height=app_cfg.height,
            buffer_size=app_cfg.camera_buffer_size,
            detector=detector,
            save_snaps=app_cfg.save_snaps,
            snaps_dir=app_cfg.snaps_dir,
        )
        cap.open()
    except Exception as exc:
        logger.error("Could not initialize detector/camera/registrar: %s", exc, exc_info=True)
        # ensure partial resources are closed
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if detector is not None:
                detector.close()
        except Exception:
            pass
        return

    window_name = config.WINDOW_NAME
    if app_cfg.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, *config.WINDOW_DEFAULT_RESOLUTION)
        except Exception:
            pass

    status_text = "Touchless Lock System Ready. Press 'r' to Register, 'q' to quit."

    registering: bool = False
    registration_detections: List[PalmDetection] = []
    registration_targets: int = config.REGISTRATION_TARGETS
    registration_handedness: Optional[str] = None

    verification_detections: List[PalmDetection] = []
    verification_targets: int = config.VERIFICATION_TARGETS
    verification_cooldown: Optional[float] = None

    frame_index: int = 0

    try:
        while True:
            frame_start = time.perf_counter()
            frame_index += 1

            now = time.time()
            # Fetch frame and detections
            success, annotated, palm_detections = False, None, None
            try:
                success, annotated, palm_detections = cap.get_palm_frame()
            except Exception as exc:
                logger.error("Error fetching frame: %s", exc, exc_info=True)
                break

            if not success or annotated is None:
                logger.warning("No frame received from source.")
                break

            # If in cooldown, skip verification processing but still display/capture frames
            if verification_cooldown is not None and now < verification_cooldown:
                remaining = verification_cooldown - now
                status_text = f"Cooldown: {remaining:.1f}s remaining"
            else:
                # Process detections
                if palm_detections:
                    logger.debug("Frame %d: %d palm detections", frame_index, len(palm_detections))
                # Registration flow
                if registering:
                    if palm_detections:
                        palm_detect = palm_detections[0]
                        registration_detections.append(palm_detect)
                        status_text = f"Registration: {len(registration_detections)}/{registration_targets} detections"
                        if len(registration_detections) >= registration_targets:
                            try:
                                user_name = input("\nRegistration complete! Enter your name: ").strip() or "Unknown"
                                success_reg, user_id = registrar.register_user_with_features(
                                    registration_detections,
                                    handedness=registration_handedness or "Right",
                                    name=user_name
                                )
                                if success_reg:
                                    status_text = f"Registration successful for {user_name}!"
                                    print_status(f"Registration successful for {user_name} ({registration_handedness or 'Right'} hand)!")
                                else:
                                    status_text = "Registration failed"
                                    print_status("Registration failed")
                            except Exception as exc:
                                logger.error("Registration error: %s", exc, exc_info=True)
                                status_text = "Registration failed"
                                print_status("Registration failed")
                            registering = False
                            registration_detections.clear()
                            registration_handedness = None
                    else:
                        status_text = "Show your palm for registration..."
                else:
                    # Normal verification flow
                    if palm_detections:
                        palm_detect = palm_detections[0]
                        landmarks = getattr(palm_detect, "landmarks", None)
                        handedness = getattr(palm_detect, "handedness", "Right")
                        palm_facing = False

                        if landmarks:
                            if callable(is_palm_facing_camera):
                                try:
                                    palm_facing = is_palm_facing_camera(landmarks)
                                except Exception:
                                    palm_facing = _fallback_is_palm_facing_camera(landmarks, handedness)
                            else:
                                palm_facing = _fallback_is_palm_facing_camera(landmarks, handedness)

                        if not palm_facing:
                            status_text = f"{handedness} palm detected but not facing camera"
                        else:
                            verification_detections.append(palm_detect)
                            status_text = f"Verification: {len(verification_detections)}/{verification_targets} detections"

                        if len(verification_detections) >= verification_targets:
                            try:
                                is_match, matched_user_id, matched_name = verify_palm_with_features(
                                    verification_detections,
                                    handedness=None,
                                    use_geometry=app_cfg.use_geometry,
                                    similarity_threshold=app_cfg.similarity_threshold
                                )
                                if is_match:
                                    display_name = matched_name if matched_name and matched_name != "Unknown" else f"User {matched_user_id}"
                                    status_text = f"Access Granted: {display_name}"
                                    print_status(f"Access Granted: {display_name}")
                                    send_esp32_signal(True, app_cfg, logger)
                                else:
                                    status_text = "Access Denied"
                                    print_status("Access Denied")
                                    send_esp32_signal(False, app_cfg, logger)
                            except Exception as exc:
                                logger.error("Verification error: %s", exc, exc_info=True)
                                status_text = "Verification failed"
                                print_status("Verification failed")
                            verification_detections.clear()
                            verification_cooldown = now + config.VERIFICATION_COOLDOWN_SECONDS
                    else:
                        status_text = "No valid palm detected. Press 'r' to Register, 'q' to quit."

            # Display
            if app_cfg.display and annotated is not None:
                try:
                    cv2.putText(annotated, status_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                except Exception:
                    logger.debug("Failed to draw status text", exc_info=True)
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0

            # Key handling
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

            # FPS limiter
            if app_cfg.max_fps is not None and app_cfg.max_fps > 0:
                target_dt = 1.0 / max(1e-6, app_cfg.max_fps)
                elapsed = time.perf_counter() - frame_start
                remaining = target_dt - elapsed
                if remaining > 0:
                    try:
                        time.sleep(remaining)
                    except Exception:
                        pass

    finally:
        # newline for console neatness
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

        # Cleanup resources gracefully
        try:
            if detector is not None:
                detector.close()
        except Exception:
            logger.debug("Error closing detector", exc_info=True)

        try:
            if cap is not None:
                cap.release()
        except Exception:
            logger.debug("Error releasing capture", exc_info=True)

        try:
            if app_cfg.display:
                cv2.destroyAllWindows()
        except Exception:
            logger.debug("Error destroying windows", exc_info=True)

        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
