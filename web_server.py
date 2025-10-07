"""
FastAPI Web Server for Palm Snapshot Collection

This module provides a FastAPI web server that:
1. Serves a web interface for palm detection
2. Collects palm snapshots via HTTP endpoints
3. Manages registration and verification modes
4. Processes palm images and returns results

Author: AI Assistant
Date: 2024
"""

import asyncio
import base64
import io
import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from detector import PalmDetector, DetectorConfig, PalmDetection
from registration import PalmRegistrar, RegistrarConfig
from verification import verify_palm_with_features
from utils import config
from utils.palm import is_palm_facing_camera

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app = FastAPI(title="Touchless Lock System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
detector: Optional[PalmDetector] = None
registrar: Optional[PalmRegistrar] = None
active_connections: List[WebSocket] = []

# --- ESP32-driven session state ---
class SessionMode(Enum):
    REGISTRATION = "registration"
    VERIFICATION = "verification"

class DeviceSession:
    def __init__(self, device_id: str, target_snapshots: int = 5):
        self.device_id = device_id
        self.mode: SessionMode = SessionMode.VERIFICATION
        self.user_name: Optional[str] = None
        self.target_snapshots: int = target_snapshots
        self.snapshots: List[PalmDetection] = []
        self.last_seen: float = time.time()
        self.last_result: Optional[Dict[str, Any]] = None
        self.cooldown_until: Optional[float] = None
        self.last_annotated_b64: Optional[str] = None
        self.recent_annotated_b64: List[str] = []

    def reset_cycle(self):
        self.snapshots.clear()
        self.cooldown_until = None

device_sessions: Dict[str, DeviceSession] = {}
last_hello_device: Optional[str] = None
SESSION_TIMEOUT_SECONDS = 60
COOLDOWN_SECONDS = 2


class SystemMode(Enum):
    IDLE = "idle"
    REGISTRATION = "registration"
    VERIFICATION = "verification"


@dataclass
class SystemState:
    mode: SystemMode = SystemMode.IDLE
    snapshots_collected: int = 0
    target_snapshots: int = 5
    snapshots: List[PalmDetection] = None
    user_name: Optional[str] = None
    handedness: Optional[str] = None
    cooldown_until: Optional[float] = None

    def __post_init__(self):
        if self.snapshots is None:
            self.snapshots = []


# Global system state
system_state = SystemState()


def initialize_components():
    """Initialize detector and registrar components."""
    global detector, registrar
    
    try:
        detector_cfg = DetectorConfig(
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        detector = PalmDetector(config=detector_cfg, max_num_hands=config.MAX_NUM_HANDS)
        
        registrar_config = RegistrarConfig(use_geometry=config.DEFAULT_USE_GEOMETRY)
        registrar = PalmRegistrar(config=registrar_config)
        
        logger.info("Components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False


def process_image(image_data: bytes) -> Dict[str, Any]:
    """Process image data and return detection results."""
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Run palm detection
        if detector is None:
            raise ValueError("Detector not initialized")
        
        annotated_image, detections = detector.detect(image)
        
        # Convert annotated image back to bytes
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_bytes = buffer.tobytes()
        annotated_b64 = base64.b64encode(annotated_bytes).decode('utf-8')
        
        return {
            "success": True,
            "detections": len(detections),
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "palm_detections": [
                {
                    "bbox": detection.bbox,
                    "handedness": detection.handedness,
                    "confidence": detection.confidence
                }
                for detection in detections
            ]
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {
            "success": False,
            "error": str(e),
            "detections": 0,
            "annotated_image": None,
            "palm_detections": []
        }


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    if not initialize_components():
        logger.error("Failed to initialize components")
        raise RuntimeError("Component initialization failed")

    # Periodic cleanup task could be added here if needed


@app.get("/")
async def root():
    """Serve the main web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Touchless Lock System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .status.idle { background-color: #f0f0f0; }
            .status.registration { background-color: #fff3cd; }
            .status.verification { background-color: #d1ecf1; }
            .status.success { background-color: #d4edda; }
            .status.error { background-color: #f8d7da; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-primary { background-color: #007bff; color: white; }
            .btn-success { background-color: #28a745; color: white; }
            .btn-warning { background-color: #ffc107; color: black; }
            .btn-danger { background-color: #dc3545; color: white; }
            #video { width: 100%; max-width: 640px; }
            #canvas { display: none; }
            .snapshot-info { margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Touchless Lock System</h1>
            
            <div id="status" class="status idle">
                Waiting for ESP32 hello... UI will enable automatically.
            </div>
            
            <div>
                <button id="startRegistration" class="btn-primary">Switch to Registration</button>
                <button id="startVerification" class="btn-success">Switch to Verification</button>
                <button id="stopProcess" class="btn-danger" style="display:none;">Stop Process</button>
            </div>
            
            <div class="snapshot-info" id="snapshotInfo" style="display:none;">
                <p>Snapshots collected: <span id="snapshotCount">0</span> / <span id="targetCount">5</span></p>
            </div>
            
            <video id="video" autoplay muted></video>
            <canvas id="canvas"></canvas>
            <div id="preview" style="margin-top:10px;"></div>
            <div id="thumbs" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;"></div>
            
            <div id="results"></div>
        </div>

        <script>
            let isProcessing = false;
            let currentMode = null;
            let uiEnabled = false;
            let activeDeviceId = null;

            // This UI is ESP32-driven; no local webcam needed now
            document.getElementById('video').style.display = 'none';

            // Event listeners
            document.getElementById('startRegistration').onclick = () => startProcess('registration');
            document.getElementById('startVerification').onclick = () => startProcess('verification');
            document.getElementById('stopProcess').onclick = stopProcess;

            async function startProcess(mode) {
                if (isProcessing) return;
                
                isProcessing = true;
                currentMode = mode;
                
                document.getElementById('startRegistration').style.display = 'none';
                document.getElementById('startVerification').style.display = 'none';
                document.getElementById('stopProcess').style.display = 'inline-block';
                document.getElementById('snapshotInfo').style.display = 'block';
                
                const response = await fetch(`/mode/${mode}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ deviceId: activeDeviceId }) });
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('status').innerHTML = `Started ${mode}. Show your palm to the camera.`;
                    document.getElementById('status').className = `status ${mode}`;
                    document.getElementById('targetCount').textContent = data.target_snapshots;
                    startSnapshotCollection();
                } else {
                    document.getElementById('status').innerHTML = `Error: ${data.error}`;
                    document.getElementById('status').className = 'status error';
                    stopProcess();
                }
            }

            async function stopProcess() {
                isProcessing = false;
                currentMode = null;
                
                document.getElementById('startRegistration').style.display = 'inline-block';
                document.getElementById('startVerification').style.display = 'inline-block';
                document.getElementById('stopProcess').style.display = 'none';
                document.getElementById('snapshotInfo').style.display = 'none';
                
                await fetch('/stop-process', { method: 'POST' });
                
                document.getElementById('status').innerHTML = 'Process stopped. System ready.';
                document.getElementById('status').className = 'status idle';
            }

            async function pollStatus() {
                try {
                    const res = await fetch('/status');
                    const s = await res.json();
                    uiEnabled = s.uiEnabled;
                    activeDeviceId = s.activeDeviceId || null;
                    document.getElementById('targetCount').textContent = s.targetSnapshots ?? 5;
                    document.getElementById('snapshotCount').textContent = s.snapshotsCollected ?? 0;
                    if (uiEnabled) {
                        document.getElementById('status').className = 'status verification';
                        document.getElementById('status').innerHTML = `ESP32 connected (device: ${activeDeviceId || 'n/a'}) - Mode: ${s.mode}`;
                        document.getElementById('snapshotInfo').style.display = 'block';
                        // Live preview
                        if (s.lastAnnotatedImage) {
                            document.getElementById('preview').innerHTML = `<img src="${s.lastAnnotatedImage}" style="max-width:640px;width:100%;border:1px solid #ccc;border-radius:4px;" />`;
                        }
                        // Thumbnails
                        const t = document.getElementById('thumbs');
                        t.innerHTML = '';
                        if (s.recentAnnotatedImages && s.recentAnnotatedImages.length) {
                            s.recentAnnotatedImages.forEach(src => {
                                const img = document.createElement('img');
                                img.src = src;
                                img.style.width = '120px';
                                img.style.border = '1px solid #ccc';
                                img.style.borderRadius = '4px';
                                t.appendChild(img);
                            });
                        }
                    } else {
                        document.getElementById('status').className = 'status idle';
                        document.getElementById('status').innerHTML = 'Waiting for ESP32 hello...';
                        document.getElementById('snapshotInfo').style.display = 'none';
                        document.getElementById('preview').innerHTML = '';
                        document.getElementById('thumbs').innerHTML = '';
                    }
                    if (s.lastResult && s.lastResult.completed) {
                        const ok = s.lastResult.ok !== false;
                        const msg = s.lastResult.message || (s.lastResult.result ? `Result: ${s.lastResult.result}` : 'Done');
                        document.getElementById('status').className = ok ? 'status success' : 'status error';
                        document.getElementById('status').innerHTML = msg;
                    }
                } catch (e) { /* ignore */ }
                setTimeout(pollStatus, 1000);
            }
            pollStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def _get_or_create_session(device_id: str) -> DeviceSession:
    global last_hello_device
    sess = device_sessions.get(device_id)
    if not sess:
        sess = DeviceSession(device_id=device_id, target_snapshots=config.VERIFICATION_TARGETS)
        device_sessions[device_id] = sess
    sess.last_seen = time.time()
    last_hello_device = device_id
    return sess


@app.post("/esp32/hello")
async def esp32_hello(body: Dict[str, Any]):
    device_id = str(body.get("deviceId", "unknown"))
    sess = _get_or_create_session(device_id)
    logger.info("ESP32 hello from %s", device_id)
    return {"ok": True, "mode": sess.mode.value, "targetSnapshots": sess.target_snapshots}


@app.post("/esp32/snapshot")
async def esp32_snapshot(body: Dict[str, Any]):
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    device_id = str(body.get("deviceId", "unknown"))
    image_data_url = body.get("image", "")
    if not image_data_url or "," not in image_data_url:
        return {"ok": False, "error": "Invalid image"}
    sess = _get_or_create_session(device_id)

    # Cooldown guard
    now_ts = time.time()
    if sess.cooldown_until and now_ts < sess.cooldown_until:
        remaining = max(0.0, sess.cooldown_until - now_ts)
        return {"ok": True, "completed": False, "cooldown": round(remaining, 2), "received": len(sess.snapshots), "target": sess.target_snapshots}

    # Decode image
    try:
        img_b64 = image_data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"ok": False, "error": "Decode failed"}
    except Exception:
        return {"ok": False, "error": "Decode failed"}

    # Detect palm and collect first valid detection
    annotated, detections = detector.detect(image)
    if detections:
        sess.snapshots.append(detections[0])

    # Encode and store annotated preview(s)
    try:
        _, buf = cv2.imencode('.jpg', annotated)
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64}"
        sess.last_annotated_b64 = data_url
        sess.recent_annotated_b64.append(data_url)
        if len(sess.recent_annotated_b64) > 5:
            sess.recent_annotated_b64 = sess.recent_annotated_b64[-5:]
    except Exception:
        pass

    received = len(sess.snapshots)
    target = sess.target_snapshots
    if received < target:
        return {"ok": True, "completed": False, "received": received, "target": target}

    # Reached target: run action
    result: Dict[str, Any]
    if sess.mode == SessionMode.REGISTRATION:
        try:
            ok, user_id = registrar.register_user_with_features(sess.snapshots, name=sess.user_name or "Unknown") if registrar else (False, None)
            if ok:
                result = {"ok": True, "completed": True, "action": "register", "result": "success", "userId": user_id, "message": f"Registration successful for {sess.user_name or 'Unknown'}"}
            else:
                result = {"ok": False, "completed": True, "action": "register", "result": "failed", "message": "Registration failed"}
        except Exception as e:
            logger.exception("Registration error: %s", e)
            result = {"ok": False, "completed": True, "action": "register", "result": "failed", "message": "Registration error"}
    else:
        try:
            is_match, matched_user_id, matched_name = verify_palm_with_features(sess.snapshots, handedness=None, use_geometry=config.DEFAULT_USE_GEOMETRY, similarity_threshold=config.DEFAULT_SIMILARITY_THRESHOLD)
            if is_match:
                result = {"ok": True, "completed": True, "action": "verify", "result": "granted", "userId": matched_user_id, "name": matched_name}
            else:
                result = {"ok": True, "completed": True, "action": "verify", "result": "denied"}
        except Exception as e:
            logger.exception("Verification error: %s", e)
            result = {"ok": False, "completed": True, "action": "verify", "result": "failed", "message": "Verification error"}

    # Save last result, reset, start cooldown
    sess.last_result = result
    sess.reset_cycle()
    sess.cooldown_until = time.time() + COOLDOWN_SECONDS
    return result


@app.post("/mode/registration")
async def set_registration_mode(body: Dict[str, Any]):
    device_id = str(body.get("deviceId", "unknown"))
    user_name = body.get("userName")
    sess = _get_or_create_session(device_id)
    sess.mode = SessionMode.REGISTRATION
    sess.user_name = user_name or sess.user_name
    sess.reset_cycle()
    sess.target_snapshots = config.REGISTRATION_TARGETS
    logger.info("Mode set to registration for %s", device_id)
    return {"ok": True, "mode": sess.mode.value, "targetSnapshots": sess.target_snapshots}


@app.post("/mode/verification")
async def set_verification_mode(body: Dict[str, Any] = {}):
    device_id = str(body.get("deviceId", last_hello_device or "unknown"))
    sess = _get_or_create_session(device_id)
    sess.mode = SessionMode.VERIFICATION
    sess.reset_cycle()
    sess.target_snapshots = config.VERIFICATION_TARGETS
    logger.info("Mode set to verification for %s", device_id)
    return {"ok": True, "mode": sess.mode.value, "targetSnapshots": sess.target_snapshots}


@app.post("/process-image")
async def process_image_endpoint(request: Dict[str, Any]):
    """Process uploaded image and handle snapshots."""
    global system_state
    
    if system_state.mode == SystemMode.IDLE:
        return {"success": False, "error": "No active process"}
    
    try:
        # Extract image data from base64
        image_data = request.get("image", "")
        if not image_data.startswith("data:image"):
            raise ValueError("Invalid image format")
        
        # Remove data URL prefix
        image_b64 = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        
        # Process the image
        result = process_image(image_bytes)
        
        if not result["success"]:
            return result
        
        # If we have palm detections, collect them
        if result["detections"] > 0 and system_state.snapshots_collected < system_state.target_snapshots:
            # For now, we'll simulate collecting snapshots
            # In a real implementation, you'd store the actual PalmDetection objects
            system_state.snapshots_collected += 1
            
            # Check if we have enough snapshots
            if system_state.snapshots_collected >= system_state.target_snapshots:
                return await complete_process()
        
        return {
            "success": True,
            "snapshots_collected": system_state.snapshots_collected,
            "target_snapshots": system_state.target_snapshots,
            "completed": system_state.snapshots_collected >= system_state.target_snapshots,
            "annotated_image": result["annotated_image"]
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"success": False, "error": str(e)}


async def complete_process():
    """Complete the current process (registration or verification)."""
    global system_state
    
    if system_state.mode == SystemMode.REGISTRATION:
        return await complete_registration()
    elif system_state.mode == SystemMode.VERIFICATION:
        return await complete_verification()
    else:
        return {"success": False, "error": "Invalid mode"}


async def complete_registration():
    """Complete registration process."""
    global system_state
    
    try:
        # In a real implementation, you would:
        # 1. Process the collected snapshots
        # 2. Extract features from each snapshot
        # 3. Register the user with the registrar
        
        # For now, simulate success
        user_name = f"User_{int(time.time())}"
        
        # Reset state
        system_state.mode = SystemMode.IDLE
        system_state.snapshots_collected = 0
        system_state.snapshots.clear()
        
        logger.info(f"Registration completed for {user_name}")
        return {
            "success": True,
            "completed": True,
            "message": f"Registration successful for {user_name}!",
            "snapshots_collected": system_state.target_snapshots
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        system_state.mode = SystemMode.IDLE
        return {
            "success": False,
            "completed": True,
            "message": f"Registration failed: {str(e)}"
        }


async def complete_verification():
    """Complete verification process."""
    global system_state
    
    try:
        # In a real implementation, you would:
        # 1. Process the collected snapshots
        # 2. Extract features from each snapshot
        # 3. Verify against registered users
        
        # For now, simulate verification
        is_match = True  # Simulate successful verification
        user_name = "Verified User"
        
        # Reset state
        system_state.mode = SystemMode.IDLE
        system_state.snapshots_collected = 0
        system_state.snapshots.clear()
        
        if is_match:
            logger.info(f"Verification successful for {user_name}")
            return {
                "success": True,
                "completed": True,
                "message": f"Access Granted: {user_name}",
                "snapshots_collected": system_state.target_snapshots
            }
        else:
            logger.info("Verification failed - no match")
            return {
                "success": False,
                "completed": True,
                "message": "Access Denied - No match found",
                "snapshots_collected": system_state.target_snapshots
            }
            
    except Exception as e:
        logger.error(f"Verification error: {e}")
        system_state.mode = SystemMode.IDLE
        return {
            "success": False,
            "completed": True,
            "message": f"Verification failed: {str(e)}"
        }


@app.get("/status")
async def get_status():
    # Determine UI enablement and active session
    now_ts = time.time()
    active_id = None
    sess_obj: Optional[DeviceSession] = None
    if last_hello_device and last_hello_device in device_sessions:
        cand = device_sessions[last_hello_device]
        if now_ts - cand.last_seen <= SESSION_TIMEOUT_SECONDS:
            active_id = last_hello_device
            sess_obj = cand
    if not active_id:
        # find any recent
        for did, sess in device_sessions.items():
            if now_ts - sess.last_seen <= SESSION_TIMEOUT_SECONDS:
                active_id = did
                sess_obj = sess
                break

    ui_enabled = sess_obj is not None
    return {
        "uiEnabled": ui_enabled,
        "activeDeviceId": active_id,
        "mode": (sess_obj.mode.value if sess_obj else SessionMode.VERIFICATION.value),
        "snapshotsCollected": (len(sess_obj.snapshots) if sess_obj else 0),
        "targetSnapshots": (sess_obj.target_snapshots if sess_obj else config.VERIFICATION_TARGETS),
        "lastResult": (sess_obj.last_result if sess_obj else None),
        "lastAnnotatedImage": (sess_obj.last_annotated_b64 if sess_obj else None),
        "recentAnnotatedImages": (sess_obj.recent_annotated_b64 if sess_obj else [])
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Send periodic status updates
            await websocket.send_json({
                "type": "status",
                "mode": system_state.mode.value,
                "snapshots_collected": system_state.snapshots_collected,
                "target_snapshots": system_state.target_snapshots
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_connections.remove(websocket)


def run_server(host: str = "0.0.0.0", port: int = 8000, network_mode: bool = False):
    """Run the FastAPI server."""
    if network_mode:
        logger.info(f"Starting Touchless Lock System server in NETWORK mode on {host}:{port}")
        logger.info(f"Server will be accessible from other devices on the network")
        logger.info(f"Local access: http://localhost:{port}")
        logger.info(f"Network access: http://{host}:{port}")
        print(f"\n🌐 Server is running!")
        print(f"📱 Local access: http://localhost:{port}")
        print(f"🌍 Network access: http://{host}:{port}")
        print(f"📋 Share this URL with other devices: http://{host}:{port}")
    else:
        logger.info(f"Starting Touchless Lock System server on {host}:{port}")
        print(f"\n🏠 Server is running!")
        print(f"📱 Open your browser: http://localhost:{port}")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
