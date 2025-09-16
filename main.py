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
)


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
    parser.add_argument("--max-fps", type=float, default=30.0, help="Limit processing FPS (e.g., 15, 30). Use 0 or negative for unlimited")
    parser.add_argument("--process-every", type=int, default=1, help="Process every Nth frame when not registering (1=every)")
    parser.add_argument("--mp-max-hands", type=int, default=2, help="Maximum number of hands to detect (mediapipe)")
    parser.add_argument("--mp-det-conf", type=float, default=0.5, help="Minimum detection confidence (mediapipe)")
    parser.add_argument("--mp-track-conf", type=float, default=0.5, help="Minimum tracking confidence (mediapipe)")
    parser.add_argument("--camera-buffer-size", type=int, default=None, help="Hint buffer size for camera capture (backend-dependent)")
    parser.add_argument("--cv2-threads", type=int, default=None, help="Set OpenCV thread count (e.g., 1 to reduce CPU)")
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
    total_targets: int = 10
    max_frames_to_try: int = 300  # allow more time; counts only when no detections
    frames_tried: int = 0
    waiting_for_movement: bool = False  # ensure one snapshot per guidance step
    last_bbox: Optional[tuple[int, int, int, int]] = None
    step_ready_at: Optional[float] = None  # 2s buffer before each snapshot
    status_text: str = "Press 'r' to Register Palm. Press 'q' to quit."
    last_capture_flash_at: Optional[float] = None
    last_announced_step_index: int = -1
    next_guide_ready_at: Optional[float] = None  # announce next guidance after this time

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
        "Return to neutral position…",
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

            if registering:
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
                                emb = extract_embedding(roi)
                                snapshots.append(Snapshot(roi_bgr=roi, bbox_xywh=bbox, landmarks_xy=det.landmarks, embedding=emb))
                                last_bbox = bbox
                                waiting_for_movement = True  # require movement before next snapshot
                                frames_tried = 0  # reset tries on success to allow more time for next movement
                                step_ready_at = None
                                last_capture_flash_at = now
                                print_line([f"Captured snapshot {len(snapshots)}/{total_targets} for '{user_id}'."])
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
                    if consistent and len(valid_embeddings_np) >= 7:
                        ok = register_user(user_id or "unknown", valid_embeddings_np)
                        if ok:
                            status_text = "Registration successful. Press 'r' to register another, 'q' to quit."
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
                    print_line([status_text, "Keys: r=register  q=quit"])
                    last_announced_step_index = -1
                    next_guide_ready_at = None
            else:
                if len(detections) > 0:
                    logger.debug("Palm detected (%d)", len(detections))
                else:
                    logger.debug("No palm detected")

            # Visual indicators (no text): yellow border during countdown, green flash after capture
            if registering:
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
                status_text = f"Starting registration for '{user_id}'. Show your palm."
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
                status_text = "Registration cancelled by user."
                print_line([status_text, "Keys: r=register  q=quit"])
                last_announced_step_index = -1
                next_guide_ready_at = None

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
