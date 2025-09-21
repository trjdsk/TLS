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
from typing import Optional, Union

import cv2
import numpy as np
import time
import sys

from camera import VideoCapture
from detector import PalmDetector, DetectionResult
from registration import register_user, verify_palm_for_registration
from verification import verify_palm


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
    max_hands: int = 1  # Focus on single hand
    detection_confidence: float = 0.3  # Lower for better detection
    tracking_confidence: float = 0.3   # Lower for better tracking
    palm_threshold: float = 0.1        # Much lower for palm validation
    # Palm verification settings
    enforce_handedness: bool = True
    mirror_correction: bool = True
    # ESP32 communication
    esp32_enabled: bool = False  # Enable ESP32 communication
    esp32_port: Optional[str] = None  # Serial port for ESP32


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_hand_side_selection() -> str:
    """Get hand side selection from user."""
    while True:
        try:
            print("\n" + "="*50)
            print("PALM REGISTRATION - HAND SIDE SELECTION")
            print("="*50)
            print("Which hand would you like to register?")
            print("1. Left hand")
            print("2. Right hand")
            print("="*50)
            
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                return "Left"
            elif choice == "2":
                return "Right"
            else:
                print("Invalid choice. Please enter 1 for Left hand or 2 for Right hand.")
        except KeyboardInterrupt:
            print("\nRegistration cancelled by user.")
            return None
        except Exception:
            print("Invalid input. Please try again.")


def get_user_name() -> str:
    """Get user name from terminal input."""
    while True:
        try:
            print("\n" + "="*50)
            print("REGISTRATION SUCCESSFUL!")
            print("="*50)
            name = input("Please enter your name: ").strip()
            
            if name:
                return name
            else:
                print("Name cannot be empty. Please enter a valid name.")
        except KeyboardInterrupt:
            print("\nUsing default name 'Unknown'.")
            return "Unknown"
        except Exception:
            print("Invalid input. Please try again.")


def get_manual_handedness(detected_handedness: Optional[str]) -> str:
    """Get manual handedness correction from user."""
    while True:
        try:
            print("\n" + "="*50)
            print("HANDEDNESS DETECTION")
            print("="*50)
            if detected_handedness:
                print(f"MediaPipe detected: {detected_handedness} hand")
                print("Is this correct?")
            else:
                print("MediaPipe could not detect handedness")
                print("Which hand are you showing?")
            print("1. Left hand")
            print("2. Right hand")
            print("="*50)
            
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                return "Left"
            elif choice == "2":
                return "Right"
            else:
                print("Invalid choice. Please enter 1 for Left hand or 2 for Right hand.")
        except KeyboardInterrupt:
            print("\nUsing detected handedness or default.")
            return detected_handedness or "Right"
        except Exception:
            print("Invalid input. Please try again.")


def correct_handedness_for_mirror(handedness: Optional[str]) -> Optional[str]:
    """Correct handedness for camera mirroring.
    
    Camera feeds are typically mirrored, so MediaPipe's handedness detection
    gets reversed. This function corrects that.
    """
    if handedness is None:
        return None
    
    if handedness == "Left":
        return "Right"
    elif handedness == "Right":
        return "Left"
    else:
        return handedness


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Touchless Lock System - Palm Detection & Verification")
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0) or stream URL")
    parser.add_argument("--width", type=int, default=None, help="Desired frame width")
    parser.add_argument("--height", type=int, default=None, help="Desired frame height")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--max-fps", type=float, default=30.0, help="Limit processing FPS (e.g., 15, 30). Use 0 or negative for unlimited")
    parser.add_argument("--max-hands", type=int, default=1, help="Maximum number of hands to detect")
    parser.add_argument("--detection-confidence", type=float, default=0.3, help="MediaPipe detection confidence threshold")
    parser.add_argument("--tracking-confidence", type=float, default=0.3, help="MediaPipe tracking confidence threshold")
    parser.add_argument("--palm-threshold", type=float, default=0.1, help="Edge Impulse palm validation threshold")
    parser.add_argument("--camera-buffer-size", type=int, default=None, help="Camera buffer size")
    parser.add_argument("--cv2-threads", type=int, default=None, help="OpenCV thread count")
    parser.add_argument("--enforce-handedness", action="store_true", default=True, help="Enforce handedness matching during verification")
    parser.add_argument("--no-enforce-handedness", action="store_false", dest="enforce_handedness", help="Disable handedness matching")
    parser.add_argument("--mirror-correction", action="store_true", default=True, help="Correct handedness for camera mirroring")
    parser.add_argument("--no-mirror-correction", action="store_false", dest="mirror_correction", help="Disable mirror correction")
    parser.add_argument("--esp32-enabled", action="store_true", help="Enable ESP32 communication")
    parser.add_argument("--esp32-port", type=str, default=None, help="ESP32 serial port (e.g., /dev/ttyUSB0)")
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
        max_hands=int(args.max_hands),
        detection_confidence=float(args.detection_confidence),
        tracking_confidence=float(args.tracking_confidence),
        palm_threshold=float(args.palm_threshold),
        enforce_handedness=getattr(args, "enforce_handedness", True),
        mirror_correction=getattr(args, "mirror_correction", True),
        esp32_enabled=getattr(args, "esp32_enabled", False),
        esp32_port=args.esp32_port,
    )


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

    # Initialize camera
    cap = VideoCapture(source=config.source, width=config.width, height=config.height, buffer_size=config.camera_buffer_size)
    try:
        cap.open()
    except Exception as exc:
        logger.error("Could not open video source: %s", exc)
        return

    # Initialize palm detector
    detector = PalmDetector(
        max_num_hands=config.max_hands,
        detection_confidence=config.detection_confidence,
        tracking_confidence=config.tracking_confidence,
        palm_threshold=config.palm_threshold,
    )

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

    def print_status(message: str) -> None:
        """Print status message to console."""
        try:
            sys.stdout.write(f"\r{message}")
            sys.stdout.flush()
        except Exception:
            pass

    def send_esp32_signal(unlock: bool) -> None:
        """Send unlock/lock signal to ESP32."""
        if not config.esp32_enabled:
            return
        
        try:
            # TODO: Implement ESP32 communication (serial/MQTT/HTTP)
            signal = "UNLOCK" if unlock else "LOCK"
            logger.info(f"ESP32 Signal: {signal}")
            print_status(f"ESP32 Signal: {signal}")
        except Exception as exc:
            logger.error(f"Failed to send ESP32 signal: {exc}")

    try:
        frame_index: int = 0
        while True:
            frame_start = time.perf_counter()
            frame_index += 1
            
            # Read frame from camera
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("No frame received from source.")
                break

            # Check verification cooldown
            now = time.time()
            if verification_cooldown is not None and now < verification_cooldown:
                remaining = verification_cooldown - now
                status_text = f"Cooldown: {remaining:.1f}s remaining"
                annotated = frame.copy()
                detections = []
                valid_detections = []
            else:
                # Run palm detection and validation
                annotated, detections = detector.detect(frame)
                
                # Process valid palm detections
                valid_detections = [d for d in detections if d.is_valid_palm and d.palm_roi is not None]
                
                # Log detection results
                if detections:
                    logger.info("Frame %d: %d total detections, %d valid palms", 
                               frame_index, len(detections), len(valid_detections))
                    for i, det in enumerate(detections):
                        logger.info("  Detection %d: bbox=%s, handedness=%s, score=%.3f, valid_palm=%s", 
                                   i, det.bbox, det.handedness, det.score, det.is_valid_palm)
            
            if registering:
                # Registration mode
                if valid_detections:
                    detection = valid_detections[0]  # Use first valid detection
                    
                    # Apply mirror correction for handedness
                    raw_handedness = detection.handedness
                    if config.mirror_correction:
                        handedness = correct_handedness_for_mirror(raw_handedness)
                    else:
                        handedness = raw_handedness
                    
                    # Store handedness on first detection
                    if registration_handedness is None:
                        registration_handedness = handedness or "Right"
                    
                    # Add to registration snapshots
                    registration_snapshots.append(detection.palm_roi)
                    status_text = f"Registration: {len(registration_snapshots)}/{registration_targets} snapshots"
                    
                    if len(registration_snapshots) >= registration_targets:
                        # Complete registration
                        try:
                            # Verify all snapshots are valid palms
                            all_valid = all(verify_palm_for_registration(roi) for roi in registration_snapshots)
                            
                            if all_valid:
                                # Extract embeddings and register user
                                from registration import extract_embedding
                                embeddings = [extract_embedding(roi) for roi in registration_snapshots]
                                
                                # Generate user ID
                                import secrets
                                user_id = "".join(secrets.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(8))
                                
                                # Get user name
                                user_name = input("\nRegistration complete! Enter your name: ").strip() or "Unknown"
                                
                                # Register user
                                success = register_user(user_id, embeddings, registration_handedness, user_name)
                                
                                if success:
                                    status_text = f"Registration successful for {user_name}!"
                                    print_status(f"Registration successful for {user_name} ({registration_handedness} hand)")
                                else:
                                    status_text = "Registration failed - duplicate user or DB error"
                                    print_status("Registration failed")
                            else:
                                status_text = "Registration failed - invalid palm detected"
                                print_status("Registration failed - invalid palm")
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
                if valid_detections:
                    detection = valid_detections[0]  # Use first valid detection
                    
                    # Apply mirror correction for handedness
                    raw_handedness = detection.handedness
                    if config.mirror_correction:
                        handedness = correct_handedness_for_mirror(raw_handedness)
                    else:
                        handedness = raw_handedness
                    
                    # Add to verification snapshots
                    verification_snapshots.append(detection.palm_roi)
                    status_text = f"Verification: {len(verification_snapshots)}/{verification_targets} snapshots"
                    
                    if len(verification_snapshots) >= verification_targets:
                        # Complete verification
                        try:
                            # Extract embeddings and verify
                            from registration import extract_embedding
                            embeddings = [extract_embedding(roi) for roi in verification_snapshots]
                            
                            # Verify against database
                            is_match, matched_user, matched_name = verify_palm(embeddings, handedness=handedness if config.enforce_handedness else None)
                            
                            if is_match:
                                display_name = matched_name if matched_name and matched_name != "Unknown" else matched_user
                                status_text = f"Access Granted: {display_name}"
                                print_status(f"Access Granted: {display_name}")
                                send_esp32_signal(True)  # Send unlock signal
                            else:
                                status_text = "Access Denied"
                                print_status("Access Denied")
                                send_esp32_signal(False)  # Send lock signal
                                
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
        
        detector.close()
        cap.release()
        
        if config.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
