"""Touchless Lock System - Main Pipeline.

Complete palm detection and verification pipeline:
1. Camera stream with palm detection
2. Edge Impulse validation of palm ROIs
3. Verification against registered users
4. ESP32 communication for unlock signals

Run:
    python main.py
Press 'q' to quit, 'r' to register new user.
"""
from __future__ import annotations

# Fix Qt Wayland display issues on Linux - MUST be set before any other imports
import os
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
from collections import deque

import cv2
import numpy as np
import time
import sys

from camera import VideoCapture
from detector import PalmDetector
from registration import register_user_with_features, verify_palm_for_registration
from verification import verify_palm_with_features


@dataclass
class AppConfig:
    source: Union[int, str] = 0
    width: Optional[int] = None
    height: Optional[int] = None
    # Performance and behavior
    max_fps: Optional[float] = 30.0
    display: bool = True
    camera_buffer_size: Optional[int] = None
    cv2_threads: Optional[int] = None
    # Detector settings
    palm_threshold: float = 0.8        # stricter default for validation
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6
    smoothing_window: int = 5
    # ESP32 communication
    esp32_enabled: bool = False  # Enable ESP32 communication
    esp32_port: Optional[str] = None  # Serial port for ESP32
    # Snapshots
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
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0) or stream URL")
    parser.add_argument("--width", type=int, default=None, help="Desired frame width")
    parser.add_argument("--height", type=int, default=None, help="Desired frame height")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--max-fps", type=float, default=30.0, help="Limit processing FPS (e.g., 15, 30). Use 0 or negative for unlimited")
    parser.add_argument("--palm-threshold", type=float, default=0.8, help="TFLite palm validation threshold (0..1)")
    parser.add_argument("--detection-confidence", type=float, default=0.6, help="MediaPipe detection confidence")
    parser.add_argument("--tracking-confidence", type=float, default=0.6, help="MediaPipe tracking confidence")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Smoothing window size for TFLite scores")
    parser.add_argument("--camera-buffer-size", type=int, default=None, help="Camera buffer size")
    parser.add_argument("--cv2-threads", type=int, default=None, help="OpenCV thread count")
    parser.add_argument("--esp32-enabled", action="store_true", help="Enable ESP32 communication")
    parser.add_argument("--esp32-port", type=str, default=None, help="ESP32 serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--save-snaps", action="store_true", help="Save detected palm snapshots to disk")
    parser.add_argument("--snaps-dir", type=str, default="snapshots", help="Directory to save snapshots")
    try:
        bool_action = argparse.BooleanOptionalAction  # Python 3.9+
        parser.add_argument("--display", action=bool_action, default=True, help="Show window and accept hotkeys")
    except Exception:
        parser.add_argument("--display", action="store_true", default=True, help="Show window and accept hotkeys")
    
    args = parser.parse_args()

    # Convert source to int if numeric
    source: Union[int, str]
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    setup_logging(args.log_level)

    # Normalize and clamp values
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
        palm_threshold=float(args.palm_threshold),
        detection_confidence=float(args.detection_confidence),
        tracking_confidence=float(args.tracking_confidence),
        smoothing_window=int(args.smoothing_window),
        esp32_enabled=getattr(args, "esp32_enabled", False),
        esp32_port=args.esp32_port,
        save_snaps=getattr(args, "save_snaps", False),
        snaps_dir=getattr(args, "snaps_dir", "snapshots"),
    )


def normalize_detection(det: Any, frame: np.ndarray) -> Dict[str, Any]:
    """
    Normalize detection to a dict with keys:
      - bbox: (x, y, w, h)
      - palm_roi: cropped ROI (grayscale) or None
      - is_valid_palm: bool
      - score: float or None
      - handedness: Optional[str]
    Supports input as dataclass-like object with attributes, or dict.
    """
    # helper accessors
    def a(obj, key, default=None):
        # try attribute then item
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    bbox = a(det, "bbox", a(det, "box", None))
    # bbox could be (x,y,w,h) or (x_min,y_min,x_max,y_max)
    palm_roi = a(det, "palm_roi", None)
    is_valid = a(det, "is_valid_palm", a(det, "is_palm", False))
    score = a(det, "tflite_score", a(det, "score", a(det, "confidence", None)))
    handedness = a(det, "handedness", None)

    # normalize bbox to (x,y,w,h)
    if bbox is None:
        # fallback - try keys in dict
        if isinstance(det, dict):
            bbox = det.get("bbox") or det.get("box") or det.get("rect")
    if bbox is None:
        return {"bbox": None, "palm_roi": None, "is_valid_palm": is_valid, "score": score, "handedness": handedness}

    # convert bbox tuple to (x,y,w,h)
    if len(bbox) == 4:
        x0, y0, a1, a2 = bbox
        # Heuristic: if values look like x_min,y_min,x_max,y_max convert, else keep as x,y,w,h
        if a1 > frame.shape[1] or a2 > frame.shape[0] or (a1 > a2 and (a1 - x0) > 0 and (a2 - y0) > 0 and a1 > x0 and a2 > y0):
            # ambiguous, but assume (x_min, y_min, x_max, y_max) if a1, a2 greater than typical width/height
            w = a1 - x0
            h = a2 - y0
            bbox_norm = (int(x0), int(y0), int(w), int(h))
        else:
            # treat as x,y,w,h
            bbox_norm = (int(x0), int(y0), int(a1), int(a2))
    else:
        # unexpected, return
        return {"bbox": None, "palm_roi": None, "is_valid_palm": is_valid, "score": score, "handedness": handedness}

    # If palm_roi missing, crop from frame (grayscale)
    if palm_roi is None and bbox_norm[2] > 0 and bbox_norm[3] > 0:
        x, y, w, h = bbox_norm
        x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
        try:
            crop = frame[y:y2, x:x2]
            if crop is None or crop.size == 0:
                palm_roi = None
            else:
                # convert to grayscale to be consistent with TFLite preprocessing
                if crop.ndim == 3 and crop.shape[2] == 3:
                    palm_roi = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                else:
                    palm_roi = crop.copy()
        except Exception:
            palm_roi = None

    return {"bbox": bbox_norm, "palm_roi": palm_roi, "is_valid_palm": bool(is_valid), "score": float(score) if score is not None else None, "handedness": handedness}


def print_status(message: str) -> None:
    """Print status message to console inline."""
    try:
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()
    except Exception:
        pass


def send_esp32_signal(unlock: bool, config: AppConfig, logger: logging.Logger) -> None:
    """Send unlock/lock signal to ESP32 (stub)."""
    if not config.esp32_enabled:
        return
    try:
        signal = "UNLOCK" if unlock else "LOCK"
        logger.info(f"ESP32 Signal: {signal}")
        print_status(f"ESP32 Signal: {signal}")
        # TODO: implement serial/MQTT/HTTP to ESP32
    except Exception as exc:
        logger.error(f"Failed to send ESP32 signal: {exc}")


def main() -> None:
    config = parse_args()
    logger = logging.getLogger("app")
    logger.info("Starting Touchless Lock System | source=%s", str(config.source))

    # OpenCV runtime tuning
    try:
        cv2.setUseOptimized(True)
        if config.cv2_threads is not None and config.cv2_threads > 0:
            cv2.setNumThreads(int(config.cv2_threads))
    except Exception:
        pass

    # Initialize palm detector
    detector = PalmDetector(
        model_path=None,
        max_num_hands=2,
        detection_confidence=config.detection_confidence,
        tracking_confidence=config.tracking_confidence,
        palm_threshold=config.palm_threshold,
        smoothing_window=config.smoothing_window
    )

    # Initialize camera with palm detector
    cap = VideoCapture(source=config.source, width=config.width, height=config.height, 
                      buffer_size=config.camera_buffer_size, detector=detector,
                      save_snaps=config.save_snaps, snaps_dir=config.snaps_dir)
    try:
        cap.open()
    except Exception as exc:
        logger.error("Could not open video source: %s", exc)
        return

    # Setup display window
    window_name = "Touchless Lock System"
    if config.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, 640, 480)
        except Exception:
            pass

    # System state
    status_text = "Touchless Lock System Ready. Press 'r' to Register, 'q' to quit."

    # Registration state
    registering: bool = False
    registration_snapshots: list[np.ndarray] = []
    registration_targets: int = 10
    registration_handedness: Optional[str] = None
    
    # Verification state
    verification_snapshots: list[np.ndarray] = []
    verification_targets: int = 5
    verification_cooldown: Optional[float] = None

    frame_index: int = 0
    try:
        while True:
            frame_start = time.perf_counter()
            frame_index += 1
            
            # Check verification cooldown
            now = time.time()
            if verification_cooldown is not None and now < verification_cooldown:
                remaining = verification_cooldown - now
                status_text = f"Cooldown: {remaining:.1f}s remaining"
                success, annotated, palm_crops = cap.get_palm_frame()
                if not success:
                    logger.warning("No frame received from source.")
                    break
            else:
                # Get frame with palm detection
                success, annotated, palm_crops = cap.get_palm_frame()
                if not success:
                    logger.warning("No frame received from source.")
                    break
                
                # Log detection results
                if palm_crops:
                    logger.info("Frame %d: %d valid palm crops detected", frame_index, len(palm_crops))
            
            if registering:
                # Registration mode
                if palm_crops:
                    # Use first palm crop
                    palm_crop = palm_crops[0]
                    
                    # Add to registration snapshots
                    registration_snapshots.append(palm_crop)
                    status_text = f"Registration: {len(registration_snapshots)}/{registration_targets} snapshots"
                    
                    if len(registration_snapshots) >= registration_targets:
                        # Complete registration
                        try:
                            # Get user name
                            user_name = input("\nRegistration complete! Enter your name: ").strip() or "Unknown"
                            
                            # Register user with features
                            success, user_id = register_user_with_features(
                                registration_snapshots, 
                                handedness=registration_handedness or "Right",
                                name=user_name,
                                feature_type="ORB"
                            )
                            
                            if success:
                                status_text = f"Registration successful for {user_name}!"
                                print_status(f"Registration successful for {user_name} ({registration_handedness or 'Right'} hand)!")
                            else:
                                status_text = "Registration failed - error occurred"
                                print_status("Registration failed")
                        except Exception as exc:
                            logger.error(f"Registration error: {exc}")
                            status_text = "Registration failed - error occurred"
                            print_status("Registration failed - error")
                        
                        # Reset registration state
                        registering = False
                        registration_snapshots.clear()
                        registration_handedness = None
                        
            else:
                # Verification mode
                if palm_crops:
                    # Use first palm crop
                    palm_crop = palm_crops[0]
                    
                    # Add to verification snapshots
                    verification_snapshots.append(palm_crop)
                    status_text = f"Verification: {len(verification_snapshots)}/{verification_targets} snapshots"
                    
                    if len(verification_snapshots) >= verification_targets:
                        # Complete verification
                        try:
                            # Verify against database using feature matching
                            is_match, matched_user_id, matched_name = verify_palm_with_features(
                                verification_snapshots,
                                handedness=None,  # Check both hands
                                feature_type="ORB",
                                match_threshold=0.15
                            )
                            
                            if is_match:
                                display_name = matched_name if matched_name and matched_name != "Unknown" else f"User {matched_user_id}"
                                status_text = f"Access Granted: {display_name}"
                                print_status(f"Access Granted: {display_name}")
                                send_esp32_signal(True, config, logger)  # Send unlock signal
                            else:
                                status_text = "Access Denied"
                                print_status("Access Denied")
                                send_esp32_signal(False, config, logger)  # Send lock signal
                                
                        except Exception as exc:
                            logger.error(f"Verification error: {exc}")
                            status_text = "Verification failed"
                            print_status("Verification failed")
                        
                        # Reset verification state and set cooldown
                        verification_snapshots.clear()
                        verification_cooldown = now + 2.0  # 2 second cooldown
                else:
                    # No valid palm detected
                    if not registering:
                        status_text = "No valid palm detected. Press 'r' to Register, 'q' to quit."

            # Display frame and handle input
            if config.display:
                # add status overlay in top-left
                try:
                    cv2.putText(annotated, status_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception:
                    pass
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0

            # Handle keyboard input
            if key == ord("q"):
                logger.info("Quit requested by user.")
                break
            elif key == ord("r") and not registering:
                # Start registration
                registering = True
                registration_snapshots.clear()
                registration_handedness = None
                status_text = "Registration started. Show your palm..."
                print_status("Registration started. Show your palm...")
            elif key == ord("c") and registering:
                # Cancel registration
                registering = False
                registration_snapshots.clear()
                registration_handedness = None
                status_text = "Registration cancelled."
                print_status("Registration cancelled.")

            # FPS limiter
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
        # Cleanup
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
