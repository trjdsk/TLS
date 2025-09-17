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
import time
import sys

from camera import VideoCapture
from detector import PalmDetector
from registration import (
    Snapshot,
    crop_roi,
    extract_embedding,
    palm_bbox_from_landmarks,
    register_user,
    validate_consistency,
    normalize_palm_orientation,
)
from verification import verify_palm


@dataclass
class AppConfig:
    backend: str = "mediapipe"
    source: Union[int, str] = 0
    width: Optional[int] = None
    height: Optional[int] = None
    immediate_forwarding: bool = True
    buffered_mode: bool = False
    palm_only: bool = True
    palm_region_only: bool = True
    # Performance and behavior
    max_fps: Optional[float] = 30.0
    process_every: int = 1
    display: bool = True
    camera_buffer_size: Optional[int] = None
    cv2_threads: Optional[int] = None
    # Detector tuning (mediapipe)
    mp_max_hands: int = 2
    mp_det_conf: float = 0.5
    mp_track_conf: float = 0.5
    # Palm verification settings
    enforce_handedness: bool = True  # Whether to enforce handedness matching during verification
    manual_handedness: bool = False  # Whether to allow manual handedness override
    mirror_correction: bool = True  # Whether to correct for camera mirroring


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
    parser = argparse.ArgumentParser(description="Palm detection with pluggable backends")
    parser.add_argument("--backend", choices=["mediapipe", "edgeimpulse"], default="mediapipe", help="Detection backend")
    parser.add_argument("--source", default="0", help="Camera index (e.g., 0) or stream URL")
    parser.add_argument("--width", type=int, default=None, help="Desired frame width")
    parser.add_argument("--height", type=int, default=None, help="Desired frame height")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--no-palm-only", action="store_false", dest="palm_only", help="Disable palm-only filtering (allow all hand poses)")
    parser.add_argument("--no-palm-region-only", action="store_false", dest="palm_region_only", help="Show full hand landmarks instead of palm region only")
    parser.add_argument("--max-fps", type=float, default=30.0, help="Limit processing FPS (e.g., 15, 30). Use 0 or negative for unlimited")
    parser.add_argument("--process-every", type=int, default=1, help="Process every Nth frame when not registering (1=every)")
    parser.add_argument("--mp-max-hands", type=int, default=2, help="Maximum number of hands to detect (mediapipe)")
    parser.add_argument("--mp-det-conf", type=float, default=0.5, help="Minimum detection confidence (mediapipe)")
    parser.add_argument("--mp-track-conf", type=float, default=0.5, help="Minimum tracking confidence (mediapipe)")
    parser.add_argument("--camera-buffer-size", type=int, default=None, help="Hint buffer size for camera capture (backend-dependent)")
    parser.add_argument("--cv2-threads", type=int, default=None, help="Set OpenCV thread count (e.g., 1 to reduce CPU)")
    parser.add_argument("--enforce-handedness", action="store_true", default=True, help="Enforce handedness matching during verification")
    parser.add_argument("--no-enforce-handedness", action="store_false", dest="enforce_handedness", help="Disable handedness matching during verification")
    parser.add_argument("--manual-handedness", action="store_true", help="Allow manual handedness override during verification")
    parser.add_argument("--mirror-correction", action="store_true", default=True, help="Correct handedness for camera mirroring (default: enabled)")
    parser.add_argument("--no-mirror-correction", action="store_false", dest="mirror_correction", help="Disable mirror correction for handedness")
    try:
        bool_action = argparse.BooleanOptionalAction  # Python 3.9+
        parser.add_argument("--immediate-forwarding", action=bool_action, default=True, help="ESP32 streams directly for detection (placeholder)")
        parser.add_argument("--buffered-mode", action=bool_action, default=False, help="ESP32 buffered SD card mode (placeholder)")
        parser.add_argument("--display", action=bool_action, default=True, help="Show window and accept hotkeys")
    except Exception:
        parser.add_argument("--immediate-forwarding", action="store_true", default=True, help="ESP32 streams directly for detection (placeholder)")
        parser.add_argument("--buffered-mode", action="store_true", default=False, help="ESP32 buffered SD card mode (placeholder)")
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

    process_every = max(1, int(args.process_every))

    return AppConfig(
        backend=args.backend,
        source=source,
        width=args.width,
        height=args.height,
        immediate_forwarding=getattr(args, "immediate_forwarding", True),
        buffered_mode=getattr(args, "buffered_mode", False),
        palm_only=bool(args.palm_only),
        palm_region_only=bool(args.palm_region_only),
        max_fps=max_fps,
        process_every=process_every,
        display=getattr(args, "display", True),
        camera_buffer_size=args.camera_buffer_size,
        cv2_threads=args.cv2_threads,
        mp_max_hands=int(args.mp_max_hands),
        mp_det_conf=float(args.mp_det_conf),
        mp_track_conf=float(args.mp_track_conf),
        enforce_handedness=getattr(args, "enforce_handedness", True),
        manual_handedness=getattr(args, "manual_handedness", False),
        mirror_correction=getattr(args, "mirror_correction", True),
    )


def main() -> None:
    config = parse_args()
    logger = logging.getLogger("app")
    logger.info("Starting palm detection | backend=%s | source=%s", config.backend, str(config.source))

    if config.buffered_mode:
        logger.info("Buffered mode requested. Placeholder only; not implemented.")
    if config.immediate_forwarding:
        logger.info("Immediate forwarding enabled (placeholder). Using local capture.")

    # OpenCV runtime tuning
    try:
        cv2.setUseOptimized(True)
        if config.cv2_threads is not None and config.cv2_threads > 0:
            cv2.setNumThreads(int(config.cv2_threads))
    except Exception:
        pass

    cap = VideoCapture(source=config.source, width=config.width, height=config.height, buffer_size=config.camera_buffer_size)
    try:
        cap.open()
    except Exception as exc:
        logger.error("Could not open video source: %s", exc)
        return

    detector = PalmDetector(
        backend=config.backend,
        palm_only=config.palm_only,
        palm_region_only=config.palm_region_only,
        max_num_hands=config.mp_max_hands,
        detection_confidence=config.mp_det_conf,
        tracking_confidence=config.mp_track_conf,
    )

    window_name = "Palm Detection"
    if config.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, 1280, 960)
        except Exception:
            pass

    # Registration state
    registering: bool = False
    user_id: Optional[str] = None
    snapshots: list[Snapshot] = []
    total_targets: int = 15
    max_frames_to_try: int = 300  # allow more time; counts only when no detections
    frames_tried: int = 0
    waiting_for_movement: bool = False  # ensure one snapshot per guidance step
    last_bbox: Optional[tuple[int, int, int, int]] = None
    step_ready_at: Optional[float] = None  # 2s buffer before each snapshot
    status_text: str = "Auto-verification active. Press 'r' to Register Palm, 'q' to quit."
    last_capture_flash_at: Optional[float] = None
    last_announced_step_index: int = -1
    next_guide_ready_at: Optional[float] = None  # announce next guidance after this time
    registration_handedness: Optional[str] = None  # Track handedness during registration

    # Verification state - auto-start on program launch
    verifying: bool = True
    verify_snapshots: list[Snapshot] = []
    verify_total_targets: int = 5
    verify_frames_tried: int = 0
    verify_settle_start: Optional[float] = None  # when palm first detected, start 2s settle timer
    verify_last_capture_flash_at: Optional[float] = None
    verify_capture_interval: float = 0.5  # 0.5s between snapshots
    verify_last_capture_time: Optional[float] = None
    verify_handedness: Optional[str] = None  # Track handedness of hand being verified
    verify_cooldown_until: Optional[float] = None  # Buffer after verification

    guidance_msgs = [
        "Hold steady…",
        "Rotate slightly to the left…",
        "Rotate slightly to the right…",
        "Move hand a bit closer…",
        "Move hand a bit further…",
        "Tilt palm up a bit…",
        "Tilt palm down a bit…",
        "Shift slightly left…",
        "Shift slightly right…",
        "Spread fingers wider…",
        "Bring fingers closer together…",
        "Tilt wrist left…",
        "Tilt wrist right…",
        "Move hand up slightly…",
        "Final steady position…",
    ]

    verify_guidance_msgs = [
        "Hold steady…",
        "Move hand slightly closer…",
        "Rotate wrist slightly…",
        "Hold steady…",
        "Hold steady…",
    ]

    def print_line(lines: list[str]) -> None:
        try:
            s = " | ".join(lines)
            sys.stdout.write(s + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    try:
        initial_help_printed: bool = False
        window_sized: bool = False
        frame_index: int = 0
        while True:
            frame_start = time.perf_counter()
            frame_index += 1
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("No frame received from source.")
                break

            # Check cooldown period before any detection processing
            now = time.time()
            if verifying and not registering and verify_cooldown_until is not None and now < verify_cooldown_until:
                remaining_cooldown = verify_cooldown_until - now
                status_text = f"Verification cooldown: {remaining_cooldown:.1f}s remaining"
                # Skip all detection during cooldown
                annotated, detections = frame, []
            else:
                # Process stride: when not registering, process every Nth frame to save CPU
                should_process = True if registering else (config.process_every <= 1 or (frame_index % config.process_every == 0))
                if should_process:
                     annotated, detections = detector.detect(frame)
                else:
                    annotated, detections = frame, []
            # After first frame, set window to the camera's native size without downscaling
            if config.display and not window_sized:
                try:
                    h0, w0 = annotated.shape[:2]
                    # Let OpenCV auto-size the window to the image; then ensure size is at least native
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 1)
                    cv2.resizeWindow(window_name, int(w0), int(h0))
                except Exception:
                    pass
                window_sized = True

            if verifying and not registering:
                # Auto-verification: simple 2s settle + 5 snapshots at 0.5s intervals
                current_count = len(verify_snapshots)
                status_text = f"Auto-verification: {current_count}/{verify_total_targets} snapshots"
                
                if len(detections) > 0:
                    verify_frames_tried = 0  # reset miss counter when palm visible
                    # Use the largest bbox if multiple detections
                    det = max(detections, key=lambda d: (d.bbox[2] * d.bbox[3]))
                    # Prefer palm-only bbox if landmarks available
                    bbox = palm_bbox_from_landmarks(det.landmarks) if det.landmarks is not None else det.bbox
                    
                    # Start settle timer when palm first detected
                    if verify_settle_start is None:
                        verify_settle_start = now
                        # Apply mirror correction if enabled
                        raw_handedness = det.handedness
                        if config.mirror_correction:
                            verify_handedness = correct_handedness_for_mirror(raw_handedness)
                        else:
                            verify_handedness = raw_handedness
                        print_line([f"Palm detected ({verify_handedness or 'Unknown'} hand). Settling for 2 seconds..."])
                    
                    # Check if settle time is complete and ready to capture
                    settle_elapsed = now - verify_settle_start
                    if settle_elapsed >= 2.0 and current_count < verify_total_targets:
                        # Check if enough time has passed since last capture
                        if verify_last_capture_time is None or (now - verify_last_capture_time) >= verify_capture_interval:
                            roi = crop_roi(frame, bbox, pad=8)
                            if roi.size > 0 and roi.shape[0] >= 16 and roi.shape[1] >= 16:
                                # Normalize palm orientation before extracting embedding
                                normalized_roi = normalize_palm_orientation(roi, det.landmarks)
                                emb = extract_embedding(normalized_roi)
                                verify_snapshots.append(Snapshot(roi_bgr=normalized_roi, bbox_xywh=bbox, landmarks_xy=det.landmarks, embedding=emb))
                                verify_last_capture_time = now
                                verify_last_capture_flash_at = now
                                print_line([f"Captured snapshot {len(verify_snapshots)}/{verify_total_targets}"])
                else:
                    # No detections; reset settle timer
                    verify_settle_start = None
                    verify_handedness = None
                    verify_frames_tried += 1  # count only when palm not detected
                
                # Check if we have enough snapshots to verify
                if len(verify_snapshots) >= verify_total_targets:
                    # Verify against database
                    valid_embeddings = [s.embedding for s in verify_snapshots if s.embedding is not None]
                    valid_embeddings_np = [e for e in valid_embeddings if isinstance(e, np.ndarray) and e.size > 0]
                    if len(valid_embeddings_np) >= 5:  # Require exactly 5 snapshots
                        # Use the captured handedness if enforcement is enabled
                        handedness = None
                        if config.enforce_handedness:
                            if config.manual_handedness:
                                # Allow manual override of handedness
                                handedness = get_manual_handedness(verify_handedness)
                            else:
                                handedness = verify_handedness
                        
                        is_match, matched_user, matched_name = verify_palm(valid_embeddings_np, handedness=handedness)
                        if is_match:
                            display_name = matched_name if matched_name and matched_name != "Unknown" else matched_user
                            status_text = f"Palm validated! User: {display_name}"
                            print_line([f"Palm validated! User: {display_name}"])
                            print_line(["Signal sent to ESP32"])
                            # Set cooldown period
                            verify_cooldown_until = now + 1.0  # 1 second buffer
                        else:
                            status_text = "Unauthorized palm!"
                            print_line(["Unauthorized palm!"])
                            # Set cooldown period for failed verification too
                            verify_cooldown_until = now + 1.0  # 1 second buffer
                    else:
                        status_text = "Verification failed — insufficient valid snapshots."
                        print_line(["Verification failed — insufficient valid snapshots."])
                    
                    # Reset for next verification cycle
                    verify_snapshots.clear()
                    verify_settle_start = None
                    verify_last_capture_time = None
                    verify_handedness = None
                    verify_cooldown_until = None  # Clear cooldown
                    print_line(["Ready for next verification..."])
                    status_text = "Auto-verification active. Press 'r' to Register Palm, 'q' to quit."
                
                # Cancel if palm disappears too long
                elif verify_frames_tried >= max_frames_to_try:
                    status_text = "Verification cancelled — palm not visible."
                    verify_snapshots.clear()
                    verify_settle_start = None
                    verify_last_capture_time = None
                    verify_handedness = None
                    verify_cooldown_until = None  # Clear cooldown
                    verify_frames_tried = 0
                    print_line(["Verification cancelled. Ready for next attempt..."])
                    status_text = "Auto-verification active. Press 'r' to Register Palm, 'q' to quit."
            elif registering:
                # Registration flow: capture up to total_targets validated snapshots
                current_count = len(snapshots)
                guide = guidance_msgs[min(current_count, len(guidance_msgs) - 1)]
                if not waiting_for_movement:
                    base_status = f"Registering '{user_id}': Capture {current_count+1}/{total_targets} — {guide}"
                else:
                    base_status = f"Adjust palm for next angle — {guide}"
                # Print the guidance for this step once when we enter a new capture step
                if not waiting_for_movement and last_announced_step_index != current_count:
                    print_line([f"Step {current_count+1}/{total_targets}: {guide}"])
                    last_announced_step_index = current_count
                if len(detections) > 0:
                    frames_tried = 0  # reset miss counter when a palm is visible
                    # Use the largest bbox if multiple detections
                    det = max(detections, key=lambda d: (d.bbox[2] * d.bbox[3]))
                    # Prefer palm-only bbox if landmarks available
                    bbox = palm_bbox_from_landmarks(det.landmarks) if det.landmarks is not None else det.bbox
                    # Compute change metrics if we're waiting for movement
                    if waiting_for_movement and last_bbox is not None:
                        lx, ly, lw, lh = last_bbox
                        cx_prev = lx + lw / 2.0
                        cy_prev = ly + lh / 2.0
                        bx, by, bw, bh = bbox
                        cx_curr = bx + bw / 2.0
                        cy_curr = by + bh / 2.0
                        centroid_shift = float(np.hypot(cx_curr - cx_prev, cy_curr - cy_prev))
                        area_prev = max(1.0, float(lw * lh))
                        area_curr = max(1.0, float(bw * bh))
                        scale_change = abs(area_curr - area_prev) / area_prev
                        # IoU for overlap
                        inter_x1 = max(lx, bx)
                        inter_y1 = max(ly, by)
                        inter_x2 = min(lx + lw, bx + bw)
                        inter_y2 = min(ly + lh, by + bh)
                        inter_w = max(0, inter_x2 - inter_x1)
                        inter_h = max(0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h
                        union = lw * lh + bw * bh - inter_area
                        iou = float(inter_area / union) if union > 0 else 0.0
                        # Consider movement sufficient if overlap reduced or centroid/scale changed enough
                        moved_enough = (iou < 0.75) or (centroid_shift > 25.0) or (scale_change > 0.15)
                        if moved_enough:
                            waiting_for_movement = False
                            frames_tried = 0
                            step_ready_at = None  # will start buffer when new pose is detected
                            # Announce next step guidance immediately
                            next_idx = len(snapshots)
                            if last_announced_step_index != next_idx:
                                next_guide = guidance_msgs[min(next_idx, len(guidance_msgs) - 1)]
                                print_line([f"Step {next_idx+1}/{total_targets}: {next_guide}"])
                                last_announced_step_index = next_idx
                    # Take a single snapshot only if not waiting_for_movement
                    if not waiting_for_movement:
                        now = time.time()
                        if step_ready_at is None:
                            step_ready_at = now + 3.0
                        # Show countdown in status
                        remaining = max(0.0, step_ready_at - now)
                        status_text = f"{base_status} | Hold steady {remaining:.1f}s"
                        if now >= step_ready_at:
                            roi = crop_roi(frame, bbox, pad=8)
                            if roi.size > 0 and roi.shape[0] >= 16 and roi.shape[1] >= 16:
                                # Capture handedness on first snapshot
                                if registration_handedness is None:
                                    raw_handedness = det.handedness or "Right"  # Default to Right if not detected
                                    # Apply mirror correction if enabled
                                    if config.mirror_correction:
                                        registration_handedness = correct_handedness_for_mirror(raw_handedness)
                                    else:
                                        registration_handedness = raw_handedness
                                
                                # Normalize palm orientation before extracting embedding
                                normalized_roi = normalize_palm_orientation(roi, det.landmarks)
                                emb = extract_embedding(normalized_roi)
                                snapshots.append(Snapshot(roi_bgr=normalized_roi, bbox_xywh=bbox, landmarks_xy=det.landmarks, embedding=emb))
                                last_bbox = bbox
                                waiting_for_movement = True  # require movement before next snapshot
                                frames_tried = 0  # reset tries on success to allow more time for next movement
                                step_ready_at = None
                                last_capture_flash_at = now
                                print_line([f"Captured snapshot {len(snapshots)}/{total_targets} for '{user_id}' ({registration_handedness} hand)."])
                                # Schedule next guidance announcement in 1 second
                                next_guide_ready_at = now + 1.0
                else:
                    # No detections; reset buffer timer while registering
                    if registering:
                        step_ready_at = None
                    status_text = base_status
                    frames_tried += 1  # count only when palm not detected

                # If still waiting for movement, announce the next step guidance after 1 second regardless of movement
                if waiting_for_movement and next_guide_ready_at is not None:
                    now2 = time.time()
                    if now2 >= next_guide_ready_at:
                        next_idx = len(snapshots)
                        if last_announced_step_index != next_idx:
                            next_guide = guidance_msgs[min(next_idx, len(guidance_msgs) - 1)]
                            print_line([f"Step {next_idx+1}/{total_targets}: {next_guide}"])
                            last_announced_step_index = next_idx
                        next_guide_ready_at = None

                # Check stopping conditions
                if len(snapshots) >= total_targets:
                    # Validate consistency: ensure they are of the same palm with slight movement
                    bboxes = [s.bbox_xywh for s in snapshots]
                    consistent = validate_consistency(bboxes, min_iou=0.10)
                    valid_embeddings = [s.embedding for s in snapshots if s.embedding is not None]

                    valid_embeddings_np = [e for e in valid_embeddings if isinstance(e, np.ndarray) and e.size > 0]
                    if consistent and len(valid_embeddings_np) >= 10:  # Require at least 10 valid embeddings out of 15
                        # Use the captured handedness
                        handedness = registration_handedness or "Right"  # Default fallback
                        
                        # Prompt for user name
                        user_name = get_user_name()
                        
                        ok = register_user(user_id or "unknown", valid_embeddings_np, handedness, user_name)
                        if ok:
                            status_text = f"Registration successful for {user_name} ({handedness} hand). Press 'r' to register another, 'q' to quit."
                        else:
                            status_text = "Registration failed — duplicate user ID or DB error."
                    else:
                        if not consistent:
                            status_text = "Registration failed — inconsistent detections (move too much or different palm)."
                        else:
                            status_text = "Registration failed — insufficient valid snapshots."
                    # Reset state regardless
                    registering = False
                    user_id = None
                    snapshots.clear()
                    frames_tried = 0
                    waiting_for_movement = False
                    last_bbox = None
                    step_ready_at = None
                    registration_handedness = None
                    print_line([status_text, "Keys: r=register  q=quit"])
                    last_announced_step_index = -1
                    next_guide_ready_at = None
                elif frames_tried >= max_frames_to_try:
                    # Too many misses; cancel (no debug folder creation)
                    status_text = "Registration cancelled — palm not visible often enough."
                    registering = False
                    user_id = None
                    snapshots.clear()
                    frames_tried = 0
                    waiting_for_movement = False
                    last_bbox = None
                    step_ready_at = None
                    registration_handedness = None
                    print_line([status_text, "Keys: r=register  q=quit"])
                    last_announced_step_index = -1
                    next_guide_ready_at = None
            else:
                if registering:
                    # Registration is active, show registration status
                    current_count = len(snapshots)
                    status_text = f"Registration: {current_count}/{total_targets} snapshots"
                elif len(detections) > 0:
                    logger.debug("Palm detected (%d)", len(detections))
                    status_text = "Palm detected. Press 'r' to register, 'q' to quit."
                else:
                    logger.debug("No palm detected")
                    status_text = "No palm detected. Press 'r' to register, 'q' to quit."

            # Visual indicators (no text): yellow border during settle, green flash after capture
            if verifying and not registering:
                now_vis = time.time()
                if verify_settle_start is not None and (now_vis - verify_settle_start) < 2.0:
                    cv2.rectangle(annotated, (2, 2), (annotated.shape[1]-3, annotated.shape[0]-3), (0, 255, 255), 2)
                if verify_last_capture_flash_at is not None and (now_vis - verify_last_capture_flash_at) < 0.3:
                    cv2.rectangle(annotated, (2, 2), (annotated.shape[1]-3, annotated.shape[0]-3), (0, 255, 0), 2)
                elif verify_last_capture_flash_at is not None and (now_vis - verify_last_capture_flash_at) >= 0.3:
                    verify_last_capture_flash_at = None
            elif registering:
                now_vis = time.time()
                if step_ready_at is not None:
                    cv2.rectangle(annotated, (2, 2), (annotated.shape[1]-3, annotated.shape[0]-3), (0, 255, 255), 2)
                if last_capture_flash_at is not None and (now_vis - last_capture_flash_at) < 0.3:
                    cv2.rectangle(annotated, (2, 2), (annotated.shape[1]-3, annotated.shape[0]-3), (0, 255, 0), 2)
                elif last_capture_flash_at is not None and (now_vis - last_capture_flash_at) >= 0.3:
                    last_capture_flash_at = None

            # Avoid per-frame terminal printing to prevent flooding
            if config.display:
                cv2.imshow(window_name, annotated)
                if not initial_help_printed:
                    print_line([status_text, "Keys: r=register  c=cancel  q=quit"])
                    initial_help_printed = True
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0

            if key == ord("q"):
                logger.info("Quit requested by user.")
                break
            if key == ord("r") and not registering:
                # Get hand side selection from user
                selected_hand = get_hand_side_selection()
                if selected_hand is None:
                    continue  # User cancelled
                
                # Auto-generate a random 16-char user_id with symbols
                try:
                    import secrets
                    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.@#$"
                    generated = "".join(secrets.choice(alphabet) for _ in range(16))
                except Exception:
                    # Fallback if secrets not available
                    import random
                    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.@#$"
                    generated = "".join(random.choice(alphabet) for _ in range(16))

                registering = True
                user_id = generated
                snapshots.clear()
                frames_tried = 0
                waiting_for_movement = False
                last_bbox = None
                step_ready_at = None
                registration_handedness = selected_hand
                status_text = f"Starting registration for '{user_id}' ({selected_hand} hand). Show your {selected_hand.lower()} palm."
                print_line([status_text, f"Snapshots: {len(snapshots)}/{total_targets}"])
                last_announced_step_index = -1
                next_guide_ready_at = None
            if key == ord("c") and registering:
                registering = False
                user_id = None
                snapshots.clear()
                frames_tried = 0
                waiting_for_movement = False
                last_bbox = None
                step_ready_at = None
                registration_handedness = None
                last_announced_step_index = -1
                next_guide_ready_at = None
                status_text = "Registration cancelled by user."
                print_line([status_text, "Keys: r=register  c=cancel  q=quit"])

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
