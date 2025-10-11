"""
FastAPI server for ESP32 device flows (hello, mode switch, snapshot ingestion).
Mirrors desktop verification behavior from `main.py` while keeping per-device
session state in memory. Targets are UI-only mirrors of device enforcement
(verification=1, registration=10).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Literal
import sqlite3

import numpy as np
import cv2
from fastapi import FastAPI, Request, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import io
import os
import time

from utils import config
from detector import PalmDetector, DetectorConfig, PalmDetection
from verification import verify_palm_with_features
from registration import PalmRegistrar, RegistrarConfig


logger = logging.getLogger("server")
if not logger.handlers:
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))


Mode = Literal["verification", "registration"]


class HelloBody(BaseModel):
    deviceId: str


class ModeVerificationBody(BaseModel):
    deviceId: str


class ModeRegistrationBody(BaseModel):
    deviceId: str
    userName: Optional[str] = None


class DeviceState(BaseModel):
    mode: Mode = "verification"
    received: int = 0
    target: int = 1
    userId: Optional[int] = None
    name: Optional[str] = None
    pendingRegistrationName: Optional[str] = None
    awaitingName: bool = False
    lastAction: Optional[str] = None      # 'verify' | 'register'
    lastResult: Optional[str] = None      # 'granted' | 'denied' | 'pending_name' | 'success' | 'failure'
    lastMessage: Optional[str] = None


VERIFICATION_TARGET = 5
REGISTRATION_TARGET = 10


app = FastAPI(title="Touchless Lock System API")


# Singletons shared across requests
_detector: Optional[PalmDetector] = None
_registrar: Optional[PalmRegistrar] = None
_devices: Dict[str, DeviceState] = {}
_device_buffers: Dict[str, List[PalmDetection]] = {}
_latest_snapshots: Dict[str, str] = {}  # device_id -> base64 encoded image
_snapshot_timers: Dict[str, float] = {}  # device_id -> last hand detection time
_snapshot_taken: Dict[str, bool] = {}  # device_id -> whether snapshot was taken
_verify_cooldown_until: Dict[str, float] = {}  # device_id -> timestamp until which verification is cooling down


def _get_detector() -> PalmDetector:
    global _detector
    if _detector is None:
        det_cfg = DetectorConfig(
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        _detector = PalmDetector(config=det_cfg, max_num_hands=config.MAX_NUM_HANDS)
    return _detector


def _get_registrar() -> PalmRegistrar:
    global _registrar
    if _registrar is None:
        reg_cfg = RegistrarConfig(use_geometry=config.DEFAULT_USE_GEOMETRY)
        _registrar = PalmRegistrar(config=reg_cfg)
    return _registrar


def _ensure_device(device_id: str) -> DeviceState:
    st = _devices.get(device_id)
    if st is None:
        st = DeviceState(mode="verification", received=0, target=VERIFICATION_TARGET)
        _devices[device_id] = st
        _device_buffers[device_id] = []
    return st


def _reset_for_mode(state: DeviceState, mode: Mode, device_id: str) -> None:
    state.mode = mode
    state.received = 0
    state.target = VERIFICATION_TARGET if mode == "verification" else REGISTRATION_TARGET
    state.userId = None
    state.name = None
    state.awaitingName = False
    _device_buffers[device_id] = []
    # Reset snapshot timers
    _snapshot_timers[device_id] = 0
    _snapshot_taken[device_id] = False


def _save_snapshots(device_id: str, frame: np.ndarray, detections: List[PalmDetection], user_name: Optional[str] = None) -> None:
    """Save snapshots with user-specific folders."""
    try:
        # Create base snapshots directory
        snaps_dir = "snapshots"
        os.makedirs(snaps_dir, exist_ok=True)
        
        # Create user-specific directory if user_name is provided
        if user_name:
            user_snaps_dir = os.path.join(snaps_dir, user_name)
            os.makedirs(user_snaps_dir, exist_ok=True)
        else:
            user_snaps_dir = snaps_dir
        
        current_time = time.time()
        ts_ms = int(current_time * 1000)
        
        for idx, det in enumerate(detections):
            x, y, w, h = det.bbox
            x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
            orig_roi = frame[y:y2, x:x2]
            hand = det.handedness or "Unknown"
            base = f"{ts_ms}_{idx}_{hand}"
            
            # Save original ROI (BGR) - this will replace any existing file
            try:
                roi_path = os.path.join(user_snaps_dir, f"roi_{base}.png")
                cv2.imwrite(roi_path, orig_roi)
                logger.debug(f"Saved original ROI snapshot: {roi_path}")
            except Exception as e:
                logger.debug(f"Failed to save original ROI snapshot: {e}")
            
            # Save 96x96 grayscale - this will replace any existing file
            try:
                roi96_path = os.path.join(user_snaps_dir, f"roi96_{base}.png")
                cv2.imwrite(roi96_path, det.palm_roi)
                logger.debug(f"Saved 96x96 ROI snapshot: {roi96_path}")
            except Exception as e:
                logger.debug(f"Failed to save 96x96 ROI snapshot: {e}")
                
    except Exception as e:
        logger.debug(f"Snapshot saving failed: {e}")


@app.post("/esp32/hello")
async def esp32_hello(body: HelloBody):
    st = _ensure_device(body.deviceId)
    # Server decides the mode and target
    if st.mode == "verification":
        st.target = VERIFICATION_TARGET
    else:
        st.target = REGISTRATION_TARGET
    
    st.received = 0
    return {
        "ok": True, 
        "mode": st.mode,
        "targetSnapshots": st.target  # Server provides target
    }


@app.post("/mode/verification")
async def set_mode_verification(body: ModeVerificationBody):
    st = _ensure_device(body.deviceId)
    _reset_for_mode(st, "verification", body.deviceId)
    return {
        "ok": True, 
        "mode": st.mode,
        "targetSnapshots": st.target
    }


@app.post("/mode/registration")
async def set_mode_registration(body: ModeRegistrationBody):
    st = _ensure_device(body.deviceId)
    _reset_for_mode(st, "registration", body.deviceId)
    st.pendingRegistrationName = (body.userName or None)
    st.awaitingName = False
    return {
        "ok": True, 
        "mode": st.mode,
        "targetSnapshots": st.target
    }


@app.post("/esp32/snapshot")
async def esp32_snapshot(
    request: Request,
    x_device_id: str = Header(alias="X-Device-ID"),
    x_snapshot_number: Optional[str] = Header(default=None, alias="X-Snapshot-Number"),
):
    # Debug: log headers
    logger.info(f"X-Device-ID: {x_device_id}, X-Snapshot-Number: {x_snapshot_number}")
    
    # Check content type from headers
    content_type = request.headers.get("content-type", "")
    if content_type != "image/jpeg":
        raise HTTPException(status_code=415, detail="Content-Type must be image/jpeg")
    
    st = _ensure_device(x_device_id)

    # Read raw JPEG data from request body
    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(status_code=400, detail="Empty snapshot body")

    # Decode JPEG -> BGR image with ESP32-CAM optimizations
    np_buf = np.frombuffer(body_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Invalid JPEG data")
    
    # ESP32-CAM specific image enhancement
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Apply sharpening to compensate for JPEG compression blur
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        # Enhance contrast for better feature extraction
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
    
    logger.debug(f"ESP32-CAM JPEG decoded and enhanced, shape: {frame.shape}, dtype: {frame.dtype}")
    
    # Store latest snapshot for UI
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        _latest_snapshots[x_device_id] = img_base64
    except Exception:
        pass  # Don't fail if image encoding fails

    # Run palm detection
    detector = _get_detector()
    try:
        _, detections = detector.detect(frame)
        logger.debug(f"Palm detection completed: {len(detections)} detections found")
    except Exception as exc:
        logger.exception("Detection failed: %s", exc)
        detections = []

    palm_detected = bool(detections)

    # Handle snapshot saving with delay mechanism
    if palm_detected:
        current_time = time.time()
        snapshot_delay = 0.5  # 0.5 seconds delay
        
        # Initialize timer if this is the first detection
        if x_device_id not in _snapshot_timers:
            _snapshot_timers[x_device_id] = current_time
            _snapshot_taken[x_device_id] = False
        
        # Check if enough time has passed and snapshot not taken yet
        if not _snapshot_taken.get(x_device_id, False) and (current_time - _snapshot_timers[x_device_id]) >= snapshot_delay:
            # Save snapshots with user name if available
            user_name = st.name if st.name else None
            _save_snapshots(x_device_id, frame, detections, user_name)
            _snapshot_taken[x_device_id] = True
            logger.debug(f"Snapshot taken for device {x_device_id} with user {user_name}")
    else:
        # No hand detected, reset the timer
        if x_device_id in _snapshot_timers:
            _snapshot_timers[x_device_id] = 0
            _snapshot_taken[x_device_id] = False

    # Cooldown gate for verification attempts
    now_ts = time.time()
    if st.mode == "verification":
        until = _verify_cooldown_until.get(x_device_id, 0.0)
        if now_ts < until:
            remaining = max(0.0, until - now_ts)
            st.lastAction = "verify"
            st.lastResult = "info"
            st.lastMessage = f"Cooldown: {remaining:.1f}s remaining"
            return JSONResponse({
                "ok": True,
                "palm_detected": palm_detected,
                "completed": False,
                "received": st.received,
                "target": st.target,
                "cooldown": remaining,
            })

    # Count only when palm_detected == True and gates pass
    if palm_detected:
        # Use the strongest/first detection
        det = detections[0]
        
        # Enhanced palm-facing validation (same as Windows version)
        palm_ok = False
        try:
            lms = getattr(det, "landmarks", None)
            if lms is not None and hasattr(lms, "landmark"):
                # Use the same strong validation as Windows version
                from utils.palm import is_palm_facing_camera
                try:
                    palm_ok = is_palm_facing_camera(lms)
                except Exception:
                    # Fallback to simplified check if main validation fails
                    p0 = np.array([lms.landmark[0].x, lms.landmark[0].y, lms.landmark[0].z])
                    p5 = np.array([lms.landmark[5].x, lms.landmark[5].y, lms.landmark[5].z])
                    p17 = np.array([lms.landmark[17].x, lms.landmark[17].y, lms.landmark[17].z])
                    v1 = p5 - p0
                    v2 = p17 - p0
                    normal = np.cross(v1, v2)
                    forward = np.array([lms.landmark[9].x, lms.landmark[9].y, lms.landmark[9].z]) - p0
                    cos_angle = float(np.dot(normal, forward)) / (float(np.linalg.norm(normal) * np.linalg.norm(forward)) + 1e-8)
                    palm_ok = (cos_angle > -0.3)
        except Exception:
            palm_ok = False

        # Enhanced quality gates (same as Windows version)
        quality_ok = False
        try:
            roi = getattr(det, "palm_roi", None)
            if roi is not None and roi.size > 0:
                h, w = roi.shape[:2]
                # More stringent quality requirements
                quality_ok = (h >= 64 and w >= 64)  # Increased from 48x48
                
                # ESP32-CAM specific quality check: variance threshold
                if quality_ok:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                    variance = np.var(gray_roi)
                    # Lower variance threshold for ESP32-CAM due to JPEG compression
                    quality_ok = variance >= 50  # Reduced from 100 for ESP32-CAM
        except Exception:
            quality_ok = False

        if palm_ok and quality_ok:
            # If registration is awaiting a name, freeze the counters to keep UI state stable
            if not (st.mode == "registration" and st.awaitingName):
                st.received += 1
                _device_buffers[x_device_id].append(det)
                logger.debug(f"Valid detection added for {x_device_id}: palm_ok={palm_ok}, quality_ok={quality_ok}")
        else:
            logger.debug(f"Detection rejected for {x_device_id}: palm_ok={palm_ok}, quality_ok={quality_ok}")

    completed = st.received >= st.target
    
    base = {
        "ok": True,
        "palm_detected": palm_detected,
        "completed": completed,
    }
    
    if completed:
        # Process completion based on mode
        try:
            buffer = _device_buffers.get(x_device_id, [])
            if st.mode == "verification":
                logger.info(f"Running verification for device {x_device_id} with {len(buffer)} detections")
                
                # ESP32-CAM optimized verification thresholds
                # Use lower similarity threshold to account for JPEG quality loss
                esp32_threshold = config.ESP32_SIMILARITY_THRESHOLD
                
                # Extract handedness from current detections for verification
                current_handedness = None
                if buffer:
                    for detection in buffer:
                        if detection.handedness and detection.handedness != "Unknown":
                            current_handedness = detection.handedness
                            break
                
                logger.info(f"HANDEDNESS DEBUG: Current detection handedness: {current_handedness}")
                
                is_match, matched_user_id, matched_name = verify_palm_with_features(
                    buffer,
                    handedness=current_handedness,
                    use_geometry=config.DEFAULT_USE_GEOMETRY,
                    similarity_threshold=esp32_threshold,
                )
                result = "granted" if is_match else "denied"
                user_id = matched_user_id if is_match else None
                name = matched_name if is_match else None
                logger.info(f"ESP32-CAM verification result for {x_device_id}: {result} (user: {name}, threshold: {esp32_threshold:.3f})")
                st.received = 0  # Reset for next cycle
                # Start a brief cooldown after verification (match desktop behavior)
                _verify_cooldown_until[x_device_id] = time.time() + config.VERIFICATION_COOLDOWN_SECONDS
                st.lastAction = "verify"
                st.lastResult = result
                st.lastMessage = (f"Access Granted: {name}" if is_match else "Access Denied")
                return JSONResponse({
                    **base,
                    "action": "verify",
                    "result": result,
                    "userId": user_id,
                    "name": name,
                })
            else:
                # Registration complete: wait for user-provided name via UI.
                # Do not reset counters; set awaitingName for UI to show modal.
                st.awaitingName = True
                st.lastAction = "register"
                st.lastResult = "pending_name"
                st.lastMessage = "Registration complete, waiting for name"
                return JSONResponse({
                    **base,
                    "action": "register",
                    "result": "pending_name",
                    "message": "Registration complete, waiting for name",
                })
        except Exception as exc:
            logger.exception("Post-processing failed: %s", exc)
            raise HTTPException(status_code=500, detail="Processing error")
    
    return JSONResponse({
        **base,
        "received": st.received,
        "target": st.target,
    })


@app.get("/")
async def root():
    return {"service": "touchless_lock_system", "status": "ok"}


@app.get("/api/debug/users")
async def debug_users():
    """Debug endpoint to check database structure and users."""
    try:
        import db
        cur = db.db.cursor()
        
        # Check what tables exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        
        # Check users table structure
        cur.execute("PRAGMA table_info(users)")
        users_schema = cur.fetchall()
        
        # Get all users
        cur.execute("SELECT * FROM users")
        all_users = cur.fetchall()
        
        return {
            "tables": [dict(t) for t in tables],
            "users_schema": [dict(s) for s in users_schema],
            "all_users": [dict(u) for u in all_users],
            "user_count": len(all_users)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug/verification")
async def debug_verification():
    """Debug endpoint to check verification configuration."""
    try:
        from utils import config
        return {
            "default_similarity_threshold": config.DEFAULT_SIMILARITY_THRESHOLD,
            "esp32_similarity_threshold": config.ESP32_SIMILARITY_THRESHOLD,
            "variance_threshold": config.VARIANCE_THRESHOLD,
            "esp32_variance_threshold": config.ESP32_VARIANCE_THRESHOLD,
            "use_geometry": config.DEFAULT_USE_GEOMETRY
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Touchless Lock System - Web UI</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; color: #333; }
            .devices { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .device-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa; }
            .device-id { font-weight: bold; font-size: 18px; color: #2c3e50; margin-bottom: 10px; }
            .status { margin: 5px 0; }
            .mode { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
            .mode.verification { background: #e3f2fd; color: #1976d2; }
            .mode.registration { background: #f3e5f5; color: #7b1fa2; }
            .progress { margin: 10px 0; }
            .progress-bar { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
            .progress-fill { height: 100%; background: linear-gradient(90deg, #4caf50, #8bc34a); transition: width 0.3s ease; }
            .snapshot { margin: 10px 0; text-align: center; }
            .snapshot img { max-width: 100%; max-height: 200px; border-radius: 4px; border: 1px solid #ddd; }
            .no-snapshot { color: #666; font-style: italic; }
            .controls { margin: 15px 0; }
            .btn { padding: 8px 16px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
            .btn-primary { background: #2196f3; color: white; }
            .btn-success { background: #4caf50; color: white; }
            .btn-warning { background: #ff9800; color: white; }
            .btn:hover { opacity: 0.8; }
            .log { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 10px; margin: 10px 0; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
            .log-entry { margin: 2px 0; }
            .log-info { color: #0066cc; }
            .log-success { color: #00aa00; }
            .log-error { color: #cc0000; }
            .refresh-btn { position: fixed; top: 20px; right: 20px; z-index: 1000; }
            .modal { display: none; position: fixed; z-index: 2000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); }
            .modal-content { background-color: #fefefe; margin: 15% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 400px; border-radius: 8px; }
            .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
            .close { color: #aaa; font-size: 28px; font-weight: bold; cursor: pointer; }
            .close:hover { color: #000; }
            .name-input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; }
            .modal-buttons { text-align: right; margin-top: 15px; }
            .header-controls { margin: 15px 0; }
            .header-controls .btn { margin: 5px; }
            .user-management { display: none; }
            .user-management.active { display: block; }
            .user-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background: #fafafa; }
            .user-info { display: flex; justify-content: space-between; align-items: center; }
            .user-details { flex: 1; }
            .user-actions { display: flex; gap: 10px; }
            .btn-danger { background: #f44336; color: white; }
            .btn-info { background: #2196f3; color: white; }
            .user-stats { font-size: 12px; color: #666; margin-top: 5px; }
            .template-list { margin-top: 10px; font-size: 12px; }
            .template-item { padding: 5px; background: #f0f0f0; margin: 2px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <button class="btn btn-primary refresh-btn" onclick="refreshData()">🔄 Refresh</button>
        <div class="container">
            <div class="header">
                <h1>🔐 Touchless Lock System</h1>
                <p>Real-time device monitoring and control</p>
                <div class="header-controls">
                    <button class="btn btn-primary" onclick="showUserManagement()">👥 User Management</button>
                    <button class="btn btn-success" onclick="refreshData()">🔄 Refresh</button>
                </div>
            </div>
            <div id="devices" class="devices">
                <div class="device-card">
                    <div class="device-id">No devices connected</div>
                    <div class="status">Waiting for ESP32 devices to connect...</div>
                </div>
            </div>
            <div class="log" id="log">
                <div class="log-entry log-info">System started. Waiting for device connections...</div>
            </div>
            
            <!-- User Management Section -->
            <div id="userManagement" class="user-management">
                <div class="header">
                    <h2>👥 User Management</h2>
                    <button class="btn btn-primary" onclick="showDevices()">← Back to Devices</button>
                </div>
                <div id="usersList">
                    <div class="log-entry log-info">Loading users...</div>
                </div>
            </div>
        </div>

        <!-- Name Input Modal -->
        <div id="nameModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Registration Complete</h3>
                    <span class="close" onclick="closeNameModal()">&times;</span>
                </div>
                <p>Please enter a name for the new user:</p>
                <input type="text" id="userNameInput" class="name-input" placeholder="Enter user name" maxlength="50">
                <div class="modal-buttons">
                    <button class="btn btn-primary" onclick="submitName()">Submit</button>
                    <button class="btn btn-warning" onclick="closeNameModal()">Cancel</button>
                </div>
            </div>
        </div>

        <script>
            let refreshInterval;
            let pendingRegistrationDevice = null;
            let currentView = 'devices'; // 'devices' or 'users'
            
            function log(message, type = 'info') {
                const logDiv = document.getElementById('log');
                const entry = document.createElement('div');
                entry.className = `log-entry log-${type}`;
                entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                logDiv.appendChild(entry);
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            async function refreshData() {
                try {
                    if (currentView === 'devices') {
                        const response = await fetch('/api/devices');
                        const data = await response.json();
                        updateDevices(data);
                    } else if (currentView === 'users') {
                        await loadUsers();
                    }
                } catch (error) {
                    log(`Error fetching data: ${error.message}`, 'error');
                }
            }
            
            async function loadUsers() {
                try {
                    log('Loading users...', 'info');
                    const response = await fetch('/api/users');
                    const data = await response.json();
                    log(`Users API response: ${JSON.stringify(data)}`, 'info');
                    updateUsers(data.users || []);
                } catch (error) {
                    log(`Error loading users: ${error.message}`, 'error');
                    document.getElementById('usersList').innerHTML = '<div class="log-entry log-error">Failed to load users</div>';
                }
            }
            
            function updateUsers(users) {
                const container = document.getElementById('usersList');
                
                if (users.length === 0) {
                    container.innerHTML = '<div class="log-entry log-info">No users registered yet</div>';
                    return;
                }
                
                container.innerHTML = users.map(user => `
                    <div class="user-card">
                        <div class="user-info">
                            <div class="user-details">
                                <div><strong>${user.name}</strong></div>
                                <div class="user-stats">
                                    ID: ${user.id} | Registered: ${new Date(user.created_at).toLocaleString()}
                                </div>
                            </div>
                            <div class="user-actions">
                                <button class="btn btn-info" onclick="showUserInfo(${user.id})">ℹ️ Info</button>
                                <button class="btn btn-danger" onclick="deleteUser(${user.id}, '${user.name}')">🗑️ Delete</button>
                            </div>
                        </div>
                    </div>
                `).join('');
            }
            
            async function showUserInfo(userId) {
                try {
                    const response = await fetch(`/api/users/${userId}/info`);
                    const data = await response.json();
                    
                    if (data.ok) {
                        const user = data.user;
                        const templatesHtml = user.templates.map(t => 
                            `<div class="template-item">
                                Template ${t.id} (${t.handedness}) - ${new Date(t.created_at).toLocaleString()} 
                                (${Math.round(t.feature_size/1024)}KB)
                            </div>`
                        ).join('');
                        
                        alert(`User Information:
Name: ${user.name}
ID: ${user.id}
Registered: ${new Date(user.created_at).toLocaleString()}
Templates: ${user.template_count}

Template Details:
${templatesHtml || 'No templates'}`);
                    } else {
                        log(`Failed to get user info: ${data.message}`, 'error');
                    }
                } catch (error) {
                    log(`Error getting user info: ${error.message}`, 'error');
                }
            }
            
            async function deleteUser(userId, userName) {
                if (!confirm(`Are you sure you want to delete user "${userName}"?\n\nThis will permanently delete the user and all their palm templates.`)) {
                    return;
                }
                
                try {
                    const response = await fetch(`/api/users/${userId}`, {
                        method: 'DELETE'
                    });
                    const data = await response.json();
                    
                    if (data.ok) {
                        log(`User "${userName}" deleted successfully`, 'success');
                        await loadUsers(); // Refresh user list
                    } else {
                        log(`Failed to delete user: ${data.message}`, 'error');
                    }
                } catch (error) {
                    log(`Error deleting user: ${error.message}`, 'error');
                }
            }
            
            function showUserManagement() {
                currentView = 'users';
                document.getElementById('devices').style.display = 'none';
                document.getElementById('userManagement').classList.add('active');
                loadUsers();
            }
            
            function showDevices() {
                currentView = 'devices';
                document.getElementById('devices').style.display = 'block';
                document.getElementById('userManagement').classList.remove('active');
                refreshData();
            }
            
            function updateDevices(devices) {
                const container = document.getElementById('devices');
                
                if (Object.keys(devices).length === 0) {
                    container.innerHTML = `
                        <div class="device-card">
                            <div class="device-id">No devices connected</div>
                            <div class="status">Waiting for ESP32 devices to connect...</div>
                        </div>
                    `;
                    return;
                }
                
                container.innerHTML = Object.entries(devices).map(([deviceId, device]) => {
                    // Show modal when server marks registration as awaiting name
                    if (device.mode === 'registration' && device.awaitingName && !pendingRegistrationDevice) {
                        pendingRegistrationDevice = deviceId;
                        setTimeout(() => showNameModal(deviceId), 300); // Small delay to let UI update
                    }
                    
                    const banner = (device.awaitingName) ? `<div class="log-entry log-info">Registration completed. Awaiting name...</div>` : '';
                    const last = (device.lastMessage) ? `<div class="log-entry ${device.lastResult === 'granted' || device.lastResult === 'success' ? 'log-success' : (device.lastResult === 'denied' || device.lastResult === 'failure' ? 'log-error' : 'log-info')}">${device.lastMessage}</div>` : '';

                    return `
                        <div class="device-card">
                            <div class="device-id">📱 ${deviceId}</div>
                            <div class="status">
                                <span class="mode ${device.mode}">${device.mode.toUpperCase()}</span>
                                <span>Received: ${device.received}/${device.target}</span>
                            </div>
                            ${banner}
                            <div class="progress">
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${(device.received / device.target) * 100}%"></div>
                                </div>
                            </div>
                            ${device.latestSnapshot ? `
                                <div class="snapshot">
                                    <img src="data:image/jpeg;base64,${device.latestSnapshot}" alt="Latest snapshot">
                                </div>
                            ` : '<div class="no-snapshot">No snapshot available</div>'}
                            ${last}
                            <div class="controls">
                                <button class="btn btn-primary" onclick="setMode('${deviceId}', 'verification')">Set Verification</button>
                                <button class="btn btn-warning" onclick="setMode('${deviceId}', 'registration')">Set Registration</button>
                            </div>
                        </div>
                    `;
                }).join('');
            }
            
            async function setMode(deviceId, mode) {
                try {
                    const response = await fetch(`/mode/${mode}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ deviceId })
                    });
                    const result = await response.json();
                    if (result.ok) {
                        log(`Device ${deviceId} set to ${mode} mode`, 'success');
                        refreshData();
                    } else {
                        log(`Failed to set mode for ${deviceId}`, 'error');
                    }
                } catch (error) {
                    log(`Error setting mode: ${error.message}`, 'error');
                }
            }
            
            // Modal functions
            function showNameModal(deviceId) {
                document.getElementById('nameModal').style.display = 'block';
                document.getElementById('userNameInput').focus();
                pendingRegistrationDevice = deviceId;
                log(`Registration completed for device ${deviceId}. Please enter a name.`, 'info');
            }
            
            function closeNameModal() {
                document.getElementById('nameModal').style.display = 'none';
                document.getElementById('userNameInput').value = '';
                pendingRegistrationDevice = null;
            }
            
            async function submitName() {
                const userName = document.getElementById('userNameInput').value.trim();
                if (!userName) {
                    alert('Please enter a name');
                    return;
                }
                
                if (!pendingRegistrationDevice) {
                    alert('No pending registration');
                    return;
                }
                
                try {
                    const response = await fetch('/api/submit-name', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            deviceId: pendingRegistrationDevice,
                            userName: userName
                        })
                    });
                    
                    const result = await response.json();
                    if (result.ok) {
                        log(`Name "${userName}" submitted for device ${pendingRegistrationDevice}`, 'success');
                        closeNameModal();
                        refreshData();
                    } else {
                        log(`Failed to submit name: ${result.message}`, 'error');
                    }
                } catch (error) {
                    log(`Error submitting name: ${error.message}`, 'error');
                }
            }
            
            // Handle Enter key in name input
            document.addEventListener('DOMContentLoaded', function() {
                const nameInput = document.getElementById('userNameInput');
                if (nameInput) {
                    nameInput.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            submitName();
                        }
                    });
                }
            });
            
            // Auto-refresh every 2 seconds
            refreshInterval = setInterval(refreshData, 2000);
            
            // Initial load
            refreshData();
            
            log('Web UI loaded. Auto-refreshing every 2 seconds.', 'info');
        </script>
    </body>
    </html>
    """


@app.get("/api/devices")
async def get_devices():
    """Get current device states and latest snapshots for UI."""
    devices = {}
    for device_id, state in _devices.items():
        devices[device_id] = {
            "deviceId": device_id,
            "mode": state.mode,
            "received": state.received,
            "target": state.target,
            "userId": state.userId,
            "name": state.name,
            "pendingRegistrationName": state.pendingRegistrationName,
            "awaitingName": state.awaitingName,
            "lastAction": state.lastAction,
            "lastResult": state.lastResult,
            "lastMessage": state.lastMessage,
            "latestSnapshot": _latest_snapshots.get(device_id)
        }
    return devices


@app.get("/api/users")
async def get_users():
    """Get all registered users."""
    try:
        import db
        cur = db.db.cursor()
        # Check if created_at column exists, if not use a default
        try:
            cur.execute("SELECT id, name, created_at FROM users ORDER BY created_at DESC")
        except sqlite3.OperationalError:
            # created_at column doesn't exist, use basic query
            cur.execute("SELECT id, name FROM users ORDER BY id DESC")
        
        users = cur.fetchall()
        # Convert to list of dicts for JSON serialization
        user_list = []
        for user in users:
            user_dict = dict(user)
            # Add created_at if it doesn't exist
            if 'created_at' not in user_dict:
                user_dict['created_at'] = 'Unknown'
            user_list.append(user_dict)
        
        return {"users": user_list}
    except Exception as e:
        logger.exception("Failed to fetch users: %s", e)
        return {"users": [], "error": str(e)}


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user and all their templates."""
    try:
        import db
        cur = db.db.cursor()
        
        # Get user info before deletion
        cur.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        user = cur.fetchone()
        if not user:
            return {"ok": False, "message": "User not found"}
        
        # Delete user templates first
        cur.execute("DELETE FROM palm_templates WHERE user_id = ?", (user_id,))
        
        # Delete user
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        db.db.commit()
        
        logger.info(f"Deleted user {user_id} ({user['name']}) and all templates")
        return {"ok": True, "message": f"User {user['name']} deleted successfully"}
        
    except Exception as e:
        logger.exception("Failed to delete user %s: %s", user_id, e)
        return {"ok": False, "message": f"Failed to delete user: {str(e)}"}


@app.get("/api/users/{user_id}/info")
async def get_user_info(user_id: int):
    """Get detailed information about a user."""
    try:
        import db
        cur = db.db.cursor()
        
        # Get user basic info - handle missing created_at column
        try:
            cur.execute("SELECT id, name, created_at FROM users WHERE id = ?", (user_id,))
        except sqlite3.OperationalError:
            cur.execute("SELECT id, name FROM users WHERE id = ?", (user_id,))
        
        user = cur.fetchone()
        if not user:
            return {"ok": False, "message": "User not found"}
        
        user_dict = dict(user)
        if 'created_at' not in user_dict:
            user_dict['created_at'] = 'Unknown'
        
        # Get template count
        cur.execute("SELECT COUNT(*) FROM palm_templates WHERE user_id = ?", (user_id,))
        template_count = cur.fetchone()[0]
        
        # Get template details - handle missing created_at column
        try:
            cur.execute("""
                SELECT id, handedness, created_at, 
                       LENGTH(features) as feature_size
                FROM palm_templates 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            """, (user_id,))
        except sqlite3.OperationalError:
            cur.execute("""
                SELECT id, handedness, LENGTH(features) as feature_size
                FROM palm_templates 
                WHERE user_id = ? 
                ORDER BY id DESC
            """, (user_id,))
        
        templates = cur.fetchall()
        template_list = []
        for template in templates:
            template_dict = dict(template)
            if 'created_at' not in template_dict:
                template_dict['created_at'] = 'Unknown'
            template_list.append(template_dict)
        
        return {
            "ok": True,
            "user": {
                "id": user_dict["id"],
                "name": user_dict["name"],
                "created_at": user_dict["created_at"],
                "template_count": template_count,
                "templates": template_list
            }
        }
        
    except Exception as e:
        logger.exception("Failed to get user info for %s: %s", user_id, e)
        return {"ok": False, "message": f"Failed to get user info: {str(e)}"}


class NameSubmissionBody(BaseModel):
    deviceId: str
    userName: str


@app.post("/api/submit-name")
async def submit_name(body: NameSubmissionBody):
    """Submit name for registration completion."""
    st = _ensure_device(body.deviceId)
    st.pendingRegistrationName = body.userName
    
    # Get the stored detections for this device
    buffer = _device_buffers.get(body.deviceId, [])
    if not buffer:
        return {"ok": False, "message": "No registration data available"}
    
    try:
        # Extract handedness from the first detection in buffer
        detected_handedness = None
        if buffer:
            for detection in buffer:
                if detection.handedness and detection.handedness != "Unknown":
                    detected_handedness = detection.handedness
                    break
        
        logger.info(f"REGISTRATION HANDEDNESS DEBUG: Detected handedness for {body.userName}: {detected_handedness}")
        
        registrar = _get_registrar()
        success, user_id = registrar.register_user_with_features(
            buffer,
            name=body.userName,
            handedness=detected_handedness,
        )
        
        if success:
            st.userId = user_id
            st.name = body.userName
            st.pendingRegistrationName = None
            st.awaitingName = False
            st.received = 0
            _device_buffers[body.deviceId] = []
            st.lastAction = "register"
            st.lastResult = "success"
            st.lastMessage = f"Registered {body.userName}"
            return {"ok": True, "message": f"Registration successful for {body.userName}", "userId": user_id}
        else:
            st.lastAction = "register"
            st.lastResult = "failure"
            st.lastMessage = "Registration failed"
            return {"ok": False, "message": "Registration failed"}
    except Exception as exc:
        logger.exception("Registration processing failed: %s", exc)
        return {"ok": False, "message": "Registration processing failed"}


