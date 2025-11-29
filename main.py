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
import socket
import logging
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict, List
import sys
import time
import math

import cv2
import requests
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
    # ESP32 watcher options
    esp32_watch: bool = True
    esp32_ip: Optional[str] = None


def setup_logging(level: str = None) -> None:
    level = level or config.LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Ensure file logging to tls.log
    try:
        logger = logging.getLogger()
        has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        if not has_file:
            fh = logging.FileHandler("tls.log")
            fh.setLevel(numeric_level)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(fh)
    except Exception:
        pass


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
    # ESP32 watcher controls (default enabled; use --no-esp32-watch to disable if supported)
    try:
        bool_action = argparse.BooleanOptionalAction
        parser.add_argument("--esp32-watch", action=bool_action, default=True, help="Run ESP32-CAM watcher (default: on)")
    except Exception:
        parser.add_argument("--esp32-watch", action="store_true", default=True, help="Run ESP32-CAM watcher (default: on)")
    parser.add_argument("--esp32-ip", type=str, default="auto", help="ESP32-CAM IP or 'auto' to derive x.y.z.184")
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
        esp32_watch=getattr(args, "esp32_watch", False),
        esp32_ip=getattr(args, "esp32_ip", None) if getattr(args, "esp32_ip", None) != "auto" else None,
    )


def _derive_device_ip_last_octet_184() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        # Fallback to common private range if detection fails
        local_ip = "192.168.0.1"
    finally:
        try:
            s.close()
        except Exception:
            pass
    parts = local_ip.split(".")
    if len(parts) == 4:
        parts[-1] = "184"
        return ".".join(parts)
    return "192.168.0.184"


_ESP_SESSION = requests.Session()
_ESP_TIMEOUT = (1.5, 2.5)


def esp_get(ip: str, path: str) -> tuple[int, Optional[Dict[str, Any]], Dict[str, Any]]:
    url = f"http://{ip}:80{path}"
    meta: Dict[str, Any] = {"url": url}
    try:
        resp = _ESP_SESSION.get(url, timeout=_ESP_TIMEOUT)
        meta["headers"] = dict(resp.headers)
        meta["ok"] = resp.ok
        try:
            data = resp.json()
        except Exception:
            data = None
        return resp.status_code, data, meta
    except requests.RequestException as exc:
        meta["error"] = str(exc)
        return 0, None, meta


def esp_heartbeat(ip: str, logger: logging.Logger) -> bool:
    """Send heartbeat to extend gate timeout by 15 seconds.
    
    Returns:
        True if heartbeat successful (gate open), False if gate closed or error
    """
    try:
        code, data, meta = esp_get(ip, "/heartbeat")
        if code == 200 and data and data.get("heartbeat") is True:
            logger.debug("Heartbeat sent successfully | gateOpen=%s", data.get("gateOpen"))
            return True
        elif code == 403:
            logger.debug("Heartbeat: gate closed")
            return False
        else:
            logger.debug("Heartbeat: unexpected response | code=%s | data=%s", code, data)
            return False
    except Exception as exc:
        logger.debug("Heartbeat failed: %s", exc)
        return False


def esp_status(ip: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Get ESP32 status including stream availability.
    
    Returns:
        Status dict with keys like 'streaming', 'irGate', 'awaitingVerification', 'cooldown', etc.
        None if request failed
    """
    try:
        code, data, meta = esp_get(ip, "/status")
        if code == 200 and data:
            logger.debug("Status check successful | streaming=%s | irGate=%s | awaitingVerification=%s | cooldown=%s",
                        data.get("streaming"), data.get("irGate"), data.get("awaitingVerification"), data.get("cooldown"))
            return data
        else:
            logger.debug("Status check failed | code=%s | data=%s", code, data)
            return None
    except Exception as exc:
        logger.debug("Status check exception: %s", exc)
        return None


def esp_verify(ip: str, result: bool, logger: logging.Logger) -> bool:
    """Send verification result, unlock servo (if successful), and close gate.
    
    When result=true:
        - Automatically unlocks the servo
        - Closes the gate
        - Response includes unlocked: true
    
    When result=false:
        - Does not unlock servo
        - Closes the gate
        - Response includes unlocked: false
    
    Args:
        ip: ESP32 IP address
        result: True for successful verification (unlocks servo), False for failed
        logger: Logger instance
        
    Returns:
        True if request successful, False otherwise
    """
    try:
        result_str = "true" if result else "false"
        code, data, meta = esp_get(ip, f"/verify?result={result_str}")
        if code == 200:
            unlocked = data.get("unlocked", False) if data else False
            logger.info("ESP32 /verify | result=%s | unlocked=%s | data=%s", result_str, unlocked, data)
            return True
        else:
            logger.warning("ESP32 /verify failed | code=%s | data=%s", code, data)
            return False
    except Exception as exc:
        logger.error("ESP32 /verify exception: %s", exc)
        return False


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

    # Resolve ESP32 IP if requested
    device_ip = app_cfg.esp32_ip or _derive_device_ip_last_octet_184()

    # Shared flag to tell watcher when main stream is active
    main_stream_active = {"value": False}
    
    # Optional ESP32-CAM watcher mode (run in background so main window can also run)
    if getattr(app_cfg, "esp32_watch", False):
        try:
            import threading, asyncio  # local import to avoid forcing in non-watcher mode
            from tls_stream_handler import StreamWatcher

            def _run_watcher() -> None:
                try:
                    watcher = StreamWatcher(device_ip)
                    watcher.main_stream_active = main_stream_active
                    asyncio.run(watcher.monitor())
                except Exception:
                    logging.getLogger("app").exception("Watcher thread crashed")

        
            t = threading.Thread(target=_run_watcher, daemon=True)
            t.start()
            logger.info("ESP32-CAM watcher started in background | ip=%s", device_ip)
        except Exception as exc:
            logger.error("Failed to start watcher: %s", exc, exc_info=True)

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

    # Prepare window immediately so UI shows while waiting for stream
    window_name = config.WINDOW_NAME
    control_panel_name = "Camera Controls"
    
    # Camera settings state
    camera_settings = {
        "rotation": 0,  # 0, 90, 180, 270 degrees
        "brightness": 0,  # -100 to 100
        "contrast": 0,  # -100 to 100
        "saturation": 0,  # -100 to 100
        "hue": 0,  # -180 to 180
        "flip_horizontal": 0,  # 0 or 1
        "flip_vertical": 0,  # 0 or 1
    }
    
    def apply_camera_settings(frame: np.ndarray) -> np.ndarray:
        """Apply camera settings transformations to frame."""
        if frame is None:
            return frame
        result = frame.copy()
        
        # Rotation
        if camera_settings["rotation"] == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif camera_settings["rotation"] == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)
        elif camera_settings["rotation"] == 270:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Flip
        if camera_settings["flip_horizontal"]:
            result = cv2.flip(result, 1)
        if camera_settings["flip_vertical"]:
            result = cv2.flip(result, 0)
        
        # Color adjustments (convert to HSV for hue/saturation, then back)
        if camera_settings["brightness"] != 0 or camera_settings["contrast"] != 0:
            alpha = 1.0 + (camera_settings["contrast"] / 100.0)
            beta = camera_settings["brightness"]
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        if camera_settings["saturation"] != 0 or camera_settings["hue"] != 0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            # Adjust saturation
            if camera_settings["saturation"] != 0:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + camera_settings["saturation"] / 100.0), 0, 255)
            # Adjust hue
            if camera_settings["hue"] != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + camera_settings["hue"]) % 180
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def on_rotation_change(val: int) -> None:
        camera_settings["rotation"] = val * 90
    
    def on_brightness_change(val: int) -> None:
        camera_settings["brightness"] = val - 100
    
    def on_contrast_change(val: int) -> None:
        camera_settings["contrast"] = val - 100
    
    def on_saturation_change(val: int) -> None:
        camera_settings["saturation"] = val - 100
    
    def on_hue_change(val: int) -> None:
        camera_settings["hue"] = val - 180
    
    def on_flip_h_change(val: int) -> None:
        camera_settings["flip_horizontal"] = val
    
    def on_flip_v_change(val: int) -> None:
        camera_settings["flip_vertical"] = val
    
    if app_cfg.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, *config.WINDOW_DEFAULT_RESOLUTION)
        except Exception:
            pass
        
        # Create control panel window
        cv2.namedWindow(control_panel_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(control_panel_name, 300, 500)
        
        # Create trackbars
        cv2.createTrackbar("Rotation (0/90/180/270)", control_panel_name, 0, 3, on_rotation_change)
        cv2.createTrackbar("Brightness", control_panel_name, 100, 200, on_brightness_change)
        cv2.createTrackbar("Contrast", control_panel_name, 100, 200, on_contrast_change)
        cv2.createTrackbar("Saturation", control_panel_name, 100, 200, on_saturation_change)
        cv2.createTrackbar("Hue", control_panel_name, 180, 360, on_hue_change)
        cv2.createTrackbar("Flip H", control_panel_name, 0, 1, on_flip_h_change)
        cv2.createTrackbar("Flip V", control_panel_name, 0, 1, on_flip_v_change)

    def show_placeholder(message: str, wait_ms: int = 100) -> int:
        """Render a placeholder frame with status + stream IP."""
        if not app_cfg.display:
            return 0
        try:
            w, h = config.WINDOW_DEFAULT_RESOLUTION
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(placeholder, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ip_label = f"Stream: {device_ip}"
            cv2.putText(placeholder, ip_label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
            cv2.imshow(window_name, placeholder)
            return cv2.waitKey(wait_ms) & 0xFF
        except Exception:
            return 0

    # Always use ESP32 webstream as source for camera pipeline
    stream_url = f"http://{device_ip}:81/stream"
    app_cfg.source = stream_url

    # Initialize detector/registrar in a background thread so GUI stays responsive
    import threading
    init_state = {"done": False, "error": None, "detector": None, "registrar": None}

    def _init_models() -> None:
        try:
            det_cfg = DetectorConfig(
                min_detection_confidence=app_cfg.detection_confidence,
                min_tracking_confidence=app_cfg.tracking_confidence,
            )
            det = PalmDetector(config=det_cfg, max_num_hands=config.MAX_NUM_HANDS)
            reg_cfg = RegistrarConfig(use_geometry=app_cfg.use_geometry)
            reg = PalmRegistrar(config=reg_cfg)
            init_state["detector"] = det
            init_state["registrar"] = reg
        except Exception as e:
            init_state["error"] = e
        finally:
            init_state["done"] = True

    threading.Thread(target=_init_models, daemon=True).start()
    
    def _heartbeat_loop() -> None:
        """Background thread that sends heartbeats every 8-10 seconds."""
        while not heartbeat_stop_event.is_set():
            try:
                esp_heartbeat(device_ip, logger)
            except Exception as exc:
                logger.debug("Heartbeat loop error: %s", exc)
            
            # Wait for interval or stop event
            if heartbeat_stop_event.wait(timeout=heartbeat_interval):
                break  # Stop event was set
    
    def start_heartbeat() -> None:
        """Start heartbeat thread if not already running."""
        nonlocal heartbeat_thread
        if heartbeat_thread is None or not heartbeat_thread.is_alive():
            heartbeat_stop_event.clear()
            heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            logger.info("Heartbeat started (interval: %.1fs)", heartbeat_interval)
    
    def stop_heartbeat() -> None:
        """Stop heartbeat thread."""
        nonlocal heartbeat_thread
        if heartbeat_thread is not None and heartbeat_thread.is_alive():
            heartbeat_stop_event.set()
            heartbeat_thread.join(timeout=1.0)
            logger.info("Heartbeat stopped")

    status_text = "Touchless Lock System Ready. Press 'r' to Register, 'q' to quit."

    registering: bool = False
    registration_detections: List[PalmDetection] = []
    registration_targets: int = config.REGISTRATION_TARGETS
    registration_handedness: Optional[str] = None
    # Delay before starting registration after a valid palm is detected
    registration_delay_start: Optional[float] = None
    registration_delay_seconds: float = 3.0

    verification_detections: List[PalmDetection] = []
    verification_targets: int = config.VERIFICATION_TARGETS
    verification_cooldown: Optional[float] = None
    # Delay before starting verification after a valid palm is detected
    recognition_delay_start: Optional[float] = None
    recognition_delay_seconds: float = 3.0
    
    # Heartbeat management
    heartbeat_thread: Optional[threading.Thread] = None
    heartbeat_stop_event = threading.Event()
    heartbeat_interval: float = 5.0  # Send heartbeat every 5 seconds
    
    # Stream reconnection cooldown (respect ESP32 cooldown period)
    stream_close_time: Optional[float] = None
    stream_reconnect_cooldown: float = 2.5  # Wait 2.5s after stream closes before reconnecting (ESP32 cooldown is 2s)

    frame_index: int = 0

    try:
        while True:
            frame_start = time.perf_counter()
            frame_index += 1

            now = time.time()
            # If models not ready yet, keep GUI responsive and show init message
            if detector is None or registrar is None:
                if init_state["done"] and init_state["error"] is not None:
                    # Initialization failed
                    key = show_placeholder("Initialization failed. Press q to quit.", wait_ms=200)
                    if key == ord("q"):
                        logger.error("Initialization failed: %s", init_state["error"], exc_info=True)
                        break
                    continue
                elif init_state["done"]:
                    detector = init_state["detector"]
                    registrar = init_state["registrar"]
                    logger.info("Recognition initialized.")
                else:
                    key = show_placeholder("Initializing recognition...", wait_ms=50)
                    if key == ord("q"):
                        logger.info("Quit requested by user during initialization.")
                        break
                    continue

            # Ensure capture is opened; if not, check status and open stream
            if cap is None:
                # Check if we're in cooldown period after stream closed
                if stream_close_time is not None:
                    elapsed = now - stream_close_time
                    if elapsed < stream_reconnect_cooldown:
                        remaining = stream_reconnect_cooldown - elapsed
                        # Show waiting screen during cooldown
                        if app_cfg.display:
                            key = show_placeholder(f"Waiting for cooldown... {remaining:.1f}s", wait_ms=100)
                            if key == ord("q"):
                                logger.info("Quit requested during cooldown.")
                                break
                        else:
                            time.sleep(0.1)
                        continue
                    else:
                        # Cooldown finished, clear the timestamp
                        stream_close_time = None
                
                # Check status endpoint to see if stream is available
                status = esp_status(device_ip, logger)
                if status is None:
                    # Status check failed, show waiting screen
                    if app_cfg.display:
                        key = show_placeholder("Checking status...", wait_ms=200)
                        if key == ord("q"):
                            logger.info("Quit requested while checking status.")
                            break
                    else:
                        time.sleep(0.2)
                    continue
                
                # Check if stream is available according to status
                ir_gate = status.get("irGate", False)
                cooldown = status.get("cooldown", False)
                cooldown_ms = status.get("cooldownMs", 0)
                
                # If in cooldown, wait for it to finish
                if cooldown and cooldown_ms > 0:
                    main_stream_active["value"] = False
                    if app_cfg.display:
                        key = show_placeholder(f"ESP32 cooldown: {cooldown_ms/1000:.1f}s", wait_ms=200)
                        if key == ord("q"):
                            logger.info("Quit requested during ESP32 cooldown.")
                            break
                    else:
                        time.sleep(0.2)
                    continue
                
                # Start stream when irGate is true
                if not ir_gate:
                    # Gate not open yet
                    main_stream_active["value"] = False
                    if app_cfg.display:
                        key = show_placeholder("Waiting for gate to open...", wait_ms=200)
                        if key == ord("q"):
                            logger.info("Quit requested while waiting for gate.")
                            break
                    else:
                        time.sleep(0.2)
                    continue
                
                # Stream is ready according to status - now open it
                # Mark stream as inactive until confirmed open
                main_stream_active["value"] = False
                
                # Attempt to open the stream while keeping window responsive (non-blocking)
                try:
                    cap = VideoCapture(
                        source=app_cfg.source,
                        width=app_cfg.width,
                        height=app_cfg.height,
                        buffer_size=app_cfg.camera_buffer_size,
                        detector=detector,
                        save_snaps=app_cfg.save_snaps,
                        snaps_dir=app_cfg.snaps_dir,
                    )
                    # Open with timeout - non-blocking
                    if cap.open(timeout=1.5):
                        logger.info("Stream opened: %s", app_cfg.source)
                        main_stream_active["value"] = True  # Mark stream as active
                        # Send immediate heartbeat to extend gate timeout
                        try:
                            esp_heartbeat(device_ip, logger)
                            logger.info("Heartbeat sent after stream opened")
                        except Exception:
                            logger.debug("Failed to send heartbeat after stream opened", exc_info=True)
                        # Start continuous heartbeat to keep gate open while stream is active
                        start_heartbeat()
                        logger.info("Heartbeat thread started to keep gate open")
                    else:
                        # Connection failed or timed out - release and retry
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        main_stream_active["value"] = False
                except Exception as exc:
                    logger.debug("Exception during stream open: %s", exc)
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
                    cap = None
                    main_stream_active["value"] = False
                
                # If still not opened, show waiting screen and keep window responsive
                if cap is None:
                    key = show_placeholder("Failed to open stream. Retrying...", wait_ms=200)

                    if key == ord("q"):
                        logger.info("Quit requested by user while waiting for stream.")
                        break
                    elif key == ord("r"):
                        # ignore registration while waiting for stream
                        pass
                    continue

            # Fetch frame and detections (non-blocking now)
            success, annotated, palm_detections = False, None, None
            stream_error = False
            try:
                # Check if stream is still valid before reading
                if cap._cap is not None:
                    if not cap._cap.isOpened():
                        stream_error = True
                    else:
                        # This is now non-blocking - it just gets the latest frame from background thread
                        success, annotated, palm_detections = cap.get_palm_frame()
                        # If no frame available, mark as error but don't block
                        if not success:
                            stream_error = True
                else:
                    stream_error = True
            except cv2.error as exc:
                logger.warning("OpenCV error (stream may be disconnected): %s", exc)
                stream_error = True
            except Exception as exc:
                logger.error("Error fetching frame: %s", exc, exc_info=True)
                stream_error = True

            if stream_error or not success or annotated is None:
                # Stream not available - check status to see what happened
                logger.debug("No frame available - checking status...")
                
                # Check status endpoint to understand what happened
                status = esp_status(device_ip, logger)
                
                # Mark stream as inactive
                main_stream_active["value"] = False
                
                # Stop heartbeat when stream disconnects
                stop_heartbeat()
                
                # Reset all active processes when stream disconnects
                if registering:
                    registering = False
                    registration_detections.clear()
                    registration_handedness = None
                    registration_delay_start = None
                    status_text = "Stream disconnected. Registration cancelled."
                
                # Reset verification state
                verification_detections.clear()
                recognition_delay_start = None
                verification_cooldown = None
                
                # Check status to see if stream should be closed
                should_close = False
                if status is not None:
                    ir_gate = status.get("irGate", False)
                    cooldown = status.get("cooldown", False)
                    
                    if not ir_gate:
                        # Gate is closed, stream should be closed
                        should_close = True
                        if cooldown:
                            logger.info("ESP32 gate closed (cooldown) - closing stream")
                        else:
                            logger.warning("ESP32 gate closed - closing stream")
                    else:
                        # Gate is still open, but we can't read frames
                        # This might be a connection issue
                        logger.warning("Gate is open but frames not readable - closing stream")
                        should_close = True
                else:
                    # Status check failed, assume stream should be closed
                    should_close = True
                
                if should_close:
                    try:
                        # Check if stream is completely dead
                        if cap is not None and cap._cap is not None and not cap._cap.isOpened():
                            logger.warning("Stream connection lost. Entering cooldown before reconnecting...")
                        else:
                            logger.warning("Stream disconnected. Entering cooldown before reconnecting...")
                        
                        # Record when stream closed for cooldown period
                        stream_close_time = now
                        try:
                            if cap is not None:
                                cap.release()
                        except Exception:
                            pass
                        cap = None
                    except Exception:
                        pass
                
                # Show waiting screen (clears the window) and keep window responsive
                if app_cfg.display:
                    key = show_placeholder("Stream disconnected. Waiting for stream...", wait_ms=50)
                    if key == ord("q"):
                        logger.info("Quit requested by user while waiting for stream.")
                        break
                else:
                    # If no display, just sleep briefly to avoid busy-waiting
                    time.sleep(0.1)
                continue
            
            # Apply camera settings transformations
            try:
                annotated = apply_camera_settings(annotated)
            except Exception as exc:
                logger.debug("Failed to apply camera settings: %s", exc)

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
                            registration_delay_start = None
                            # Stop heartbeat if palm not facing
                            if len(registration_detections) == 0:
                                stop_heartbeat()
                        else:
                            # Start heartbeat immediately when valid palm is detected (before delay)
                            if registration_delay_start is None:
                                registration_delay_start = now
                                start_heartbeat()  # Start heartbeat as soon as valid palm detected
                            
                            # Start or continue the 3s delay before registration
                            elapsed = now - (registration_delay_start or now)
                            if elapsed < registration_delay_seconds:
                                remaining = max(0.0, registration_delay_seconds - elapsed)
                                status_text = f"Hold steady... starting in {remaining:.1f}s"
                            else:
                                # Delay passed, start collecting detections
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
                            
                            # Send verification result (heartbeat continues until stream closes)
                            try:
                                esp_verify(device_ip, success_reg, logger)
                                # Gate will close, set cooldown timestamp
                                stream_close_time = now
                            except Exception:
                                logger.debug("ESP32 /verify call failed after registration", exc_info=True)

                            registering = False
                            registration_detections.clear()
                            registration_handedness = None
                            registration_delay_start = None
                            # Enter 10s cooldown after registration completes
                            verification_cooldown = now + 10.0
                    else:
                        status_text = "Show your palm for registration..."
                        registration_delay_start = None
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
                            recognition_delay_start = None
                            # Stop heartbeat if palm not facing
                            if len(verification_detections) == 0:
                                stop_heartbeat()
                        else:
                            # Start heartbeat immediately when valid palm is detected (before delay)
                            if recognition_delay_start is None:
                                recognition_delay_start = now
                                start_heartbeat()  # Start heartbeat as soon as valid palm detected
                            
                            # Start or continue the 3s delay before recognition
                            elapsed = now - (recognition_delay_start or now)
                            if elapsed < recognition_delay_seconds:
                                remaining = max(0.0, recognition_delay_seconds - elapsed)
                                status_text = f"Hold steady... starting in {remaining:.1f}s"
                            else:
                                # Delay passed, start collecting detections
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
                                # Send verification result (heartbeat continues until stream closes)
                                if is_match:
                                    display_name = matched_name if matched_name and matched_name != "Unknown" else f"User {matched_user_id}"
                                    status_text = f"Access Granted: {display_name}"
                                    print_status(f"Access Granted: {display_name}")
                                    # Successful verification: send verify=true
                                    try:
                                        esp_verify(device_ip, True, logger)
                                        # Gate will close, set cooldown timestamp
                                        stream_close_time = now
                                    except Exception:
                                        logger.debug("ESP32 /verify call failed after successful verification", exc_info=True)
                                else:
                                    status_text = "Access Denied"
                                    print_status("Access Denied")
                                    # Failed verification: send verify=false
                                    try:
                                        esp_verify(device_ip, False, logger)
                                        # Gate will close, set cooldown timestamp
                                        stream_close_time = now
                                    except Exception:
                                        logger.debug("ESP32 /verify call failed after failed verification", exc_info=True)
                            except Exception as exc:
                                logger.error("Verification error: %s", exc, exc_info=True)
                                status_text = "Verification failed"
                                print_status("Verification failed")
                            verification_detections.clear()
                            recognition_delay_start = None
                            # Enter 10s cooldown after verification completes
                            verification_cooldown = now + 10.0
                    else:
                        status_text = "No valid palm detected. Press 'r' to Register, 'q' to quit."
                        recognition_delay_start = None
                        # Stop heartbeat if no palm detected (user moved away)
                        if len(verification_detections) > 0:
                            stop_heartbeat()
                            verification_detections.clear()

            # Display
            if app_cfg.display and annotated is not None:
                try:
                    cv2.putText(annotated, status_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    ip_label = f"Stream: {device_ip}"
                    cv2.putText(annotated, ip_label, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (180, 255, 180), 1)
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
                registration_delay_start = None
                status_text = "Registration started. Show your palm..."
                print_status(status_text)
            elif key == ord("c") and registering:
                registering = False
                registration_detections.clear()
                registration_handedness = None
                registration_delay_start = None
                stop_heartbeat()
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
        
        # Stop heartbeat on shutdown
        try:
            stop_heartbeat()
        except Exception:
            pass

        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
