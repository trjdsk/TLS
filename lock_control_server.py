#!/usr/bin/env python3
"""
Web server for manual ESP32 lock/unlock control.
Provides a simple web interface to control the lock after verifying user identity.
"""

import argparse
import logging
import socket
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import requests
from typing import Optional, Dict, Any, Tuple
from db import Database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# ESP32 configuration
ESP32_PORT = 80
ESP32_TIMEOUT = (1.5, 2.5)

# Global ESP32 IP (set when app is created)
_esp32_ip: Optional[str] = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lock Control - TLS</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 500px;
            width: 100%;
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
            font-weight: 600;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .status-section {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .status-item:last-child {
            border-bottom: none;
        }
        
        .status-label {
            color: #666;
            font-size: 14px;
            font-weight: 500;
        }
        
        .status-value {
            color: #333;
            font-size: 14px;
            font-weight: 600;
        }
        
        .status-value.locked {
            color: #dc3545;
        }
        
        .status-value.unlocked {
            color: #28a745;
        }
        
        .status-value.unknown {
            color: #6c757d;
        }
        
        .user-input-section {
            margin-bottom: 30px;
        }
        
        label {
            display: block;
            color: #333;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 20px;
        }
        
        button {
            padding: 14px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-lock {
            background: #dc3545;
            color: white;
        }
        
        .btn-lock:hover:not(:disabled) {
            background: #c82333;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4);
        }
        
        .btn-unlock {
            background: #28a745;
            color: white;
        }
        
        .btn-unlock:hover:not(:disabled) {
            background: #218838;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
        }
        
        .btn-refresh {
            background: #667eea;
            color: white;
            grid-column: 1 / -1;
        }
        
        .btn-refresh:hover:not(:disabled) {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .message {
            margin-top: 20px;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            display: none;
        }
        
        .message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .message.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 8px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .esp32-info {
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #004085;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 Lock Control</h1>
        <p class="subtitle">Touchless Lock System - Manual Control</p>
        
        <div class="esp32-info">
            <strong>ESP32:</strong> <span id="esp32-ip">{{ esp32_ip }}</span>
        </div>
        
        <div class="status-section">
            <div class="status-item">
                <span class="status-label">Lock Status:</span>
                <span class="status-value unknown" id="lock-status">Unknown</span>
            </div>
            <div class="status-item">
                <span class="status-label">ESP32 Status:</span>
                <span class="status-value unknown" id="esp32-status">Checking...</span>
            </div>
        </div>
        
        <div class="user-input-section">
            <label for="user-name">Enter Your Name:</label>
            <input type="text" id="user-name" placeholder="Your registered name" autocomplete="off">
        </div>
        
        <div class="button-group">
            <button class="btn-lock" id="btn-lock" onclick="controlLock('lock')">🔒 Lock</button>
            <button class="btn-unlock" id="btn-unlock" onclick="controlLock('unlock')">🔓 Unlock</button>
            <button class="btn-refresh" id="btn-refresh" onclick="checkStatus()">🔄 Refresh Status</button>
        </div>
        
        <div class="message" id="message"></div>
    </div>
    
    <script>
        let currentStatus = null;
        
        // Check status on page load
        window.addEventListener('DOMContentLoaded', () => {
            checkStatus();
        });
        
        async function checkStatus() {
            const statusEl = document.getElementById('esp32-status');
            const lockStatusEl = document.getElementById('lock-status');
            const messageEl = document.getElementById('message');
            const btnRefresh = document.getElementById('btn-refresh');
            
            statusEl.textContent = 'Checking...';
            statusEl.className = 'status-value unknown';
            lockStatusEl.textContent = 'Unknown';
            lockStatusEl.className = 'status-value unknown';
            btnRefresh.disabled = true;
            
            try {
                const response = await fetch('/api/check-status', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    statusEl.textContent = 'Online';
                    statusEl.className = 'status-value unlocked';
                    
                    if (data.status && data.status.locked !== undefined) {
                        currentStatus = data.status;
                        lockStatusEl.textContent = data.status.locked ? 'Locked' : 'Unlocked';
                        lockStatusEl.className = data.status.locked ? 'status-value locked' : 'status-value unlocked';
                    } else {
                        lockStatusEl.textContent = 'Unknown';
                        lockStatusEl.className = 'status-value unknown';
                    }
                    
                    showMessage('Status updated successfully', 'success');
                } else {
                    statusEl.textContent = 'Offline';
                    statusEl.className = 'status-value locked';
                    lockStatusEl.textContent = 'Unknown';
                    lockStatusEl.className = 'status-value unknown';
                    showMessage(data.error || 'Failed to check status', 'error');
                }
            } catch (error) {
                statusEl.textContent = 'Error';
                statusEl.className = 'status-value locked';
                lockStatusEl.textContent = 'Unknown';
                lockStatusEl.className = 'status-value unknown';
                showMessage('Network error: ' + error.message, 'error');
            } finally {
                btnRefresh.disabled = false;
            }
        }
        
        async function controlLock(action) {
            const userName = document.getElementById('user-name').value.trim();
            const messageEl = document.getElementById('message');
            const btnLock = document.getElementById('btn-lock');
            const btnUnlock = document.getElementById('btn-unlock');
            
            if (!userName) {
                showMessage('Please enter your name', 'error');
                return;
            }
            
            // Disable buttons
            btnLock.disabled = true;
            btnUnlock.disabled = true;
            
            try {
                // First verify user
                const verifyResponse = await fetch('/api/verify-user', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ name: userName })
                });
                
                const verifyData = await verifyResponse.json();
                
                if (!verifyData.success) {
                    showMessage(verifyData.error || 'User verification failed', 'error');
                    btnLock.disabled = false;
                    btnUnlock.disabled = false;
                    return;
                }
                
                // Check status before action
                await checkStatus();
                
                // Perform lock action
                const lockResponse = await fetch('/api/lock', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: action })
                });
                
                const lockData = await lockResponse.json();
                
                if (lockData.success) {
                    const actionText = action === 'lock' ? 'locked' : 'unlocked';
                    showMessage(`Lock ${actionText} successfully`, 'success');
                    
                    // Update status
                    setTimeout(() => {
                        checkStatus();
                    }, 500);
                } else {
                    showMessage(lockData.error || 'Failed to control lock', 'error');
                }
            } catch (error) {
                showMessage('Network error: ' + error.message, 'error');
            } finally {
                btnLock.disabled = false;
                btnUnlock.disabled = false;
            }
        }
        
        function showMessage(text, type) {
            const messageEl = document.getElementById('message');
            messageEl.textContent = text;
            messageEl.className = `message ${type}`;
            messageEl.style.display = 'block';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                messageEl.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
"""


def _derive_device_ip_last_octet_184() -> str:
    """Derive ESP32 IP by replacing last octet with 184."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.split(".")
        if len(parts) == 4:
            parts[-1] = "184"
            return ".".join(parts)
    except Exception:
        pass
    return "192.168.0.184"


def check_esp32_status(esp32_ip: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check ESP32 lock status."""
    ip = esp32_ip or _esp32_ip
    if not ip:
        return False, None
    
    try:
        url = f"http://{ip}:{ESP32_PORT}/lock?action=status"
        resp = requests.get(url, timeout=ESP32_TIMEOUT)
        
        # Try to parse response even if status is not 200
        try:
            data = resp.json()
        except Exception:
            data = None
        
        if resp.status_code == 200:
            if data and data.get("ok"):
                return True, data
            else:
                logger.debug("ESP32 status check returned non-ok: %s", data)
        else:
            logger.warning("ESP32 status check returned status %d: %s", resp.status_code, data)
        return False, None
    except requests.RequestException as e:
        logger.debug("ESP32 status check failed: %s", e)
        return False, None
    except Exception as e:
        logger.exception("Unexpected error checking ESP32 status: %s", e)
        return False, None


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template_string(HTML_TEMPLATE, esp32_ip=_esp32_ip or "Not configured")


@app.route('/api/check-status', methods=['POST'])
def api_check_status():
    """API endpoint to check ESP32 status."""
    success, status = check_esp32_status(_esp32_ip)
    
    if success:
        return jsonify({
            'success': True,
            'status': status
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to connect to ESP32 or get status'
        }), 500


@app.route('/api/verify-user', methods=['POST'])
def api_verify_user():
    """API endpoint to verify user name against database."""
    data = request.get_json()
    name = data.get('name', '').strip() if data else ''
    
    if not name:
        return jsonify({
            'success': False,
            'error': 'Name is required'
        }), 400
    
    # Create a new database connection for this thread/request
    # SQLite connections are not thread-safe, so we need a new one per thread
    web_db = Database()
    try:
        with web_db.transaction() as cur:
            cur.execute("SELECT id FROM users WHERE name = ?", (name,))
            row = cur.fetchone()
            user_id = row["id"] if row else None
        
        if user_id:
            return jsonify({
                'success': True,
                'user_id': user_id,
                'name': name
            })
        else:
            return jsonify({
                'success': False,
                'error': 'User not found. Please register first.'
            }), 404
    except Exception as e:
        logger.exception("Error verifying user: %s", e)
        return jsonify({
            'success': False,
            'error': 'Database error'
        }), 500
    finally:
        # Close the database connection for this thread
        try:
            web_db.close()
        except Exception:
            pass


@app.route('/api/lock', methods=['POST'])
def api_lock():
    """API endpoint to control the lock."""
    data = request.get_json()
    action = data.get('action', '').lower() if data else ''
    
    if action not in ['lock', 'unlock', 'toggle']:
        return jsonify({
            'success': False,
            'error': 'Invalid action. Use: lock, unlock, or toggle'
        }), 400
    
    if not _esp32_ip:
        return jsonify({
            'success': False,
            'error': 'ESP32 IP not configured'
        }), 500
    
    try:
        url = f"http://{_esp32_ip}:{ESP32_PORT}/lock?action={action}"
        logger.debug("Sending lock control request to ESP32: %s", url)
        resp = requests.get(url, timeout=ESP32_TIMEOUT)
        
        # Try to parse response body even if status is not 200
        try:
            response_data = resp.json()
        except Exception:
            response_data = None
        
        logger.debug("ESP32 response: status=%d, data=%s", resp.status_code, response_data)
        
        if resp.status_code == 200:
            if response_data and response_data.get("ok"):
                return jsonify({
                    'success': True,
                    'locked': response_data.get('locked'),
                    'toggled': response_data.get('toggled')
                })
            else:
                error_msg = response_data.get('error', 'Lock action failed') if response_data else 'Lock action failed'
                logger.warning("ESP32 lock action failed: %s", error_msg)
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 400
        else:
            # ESP32 returned an error status
            error_msg = f'ESP32 returned status {resp.status_code}'
            if response_data:
                if 'error' in response_data:
                    error_msg = response_data.get('error', error_msg)
                elif 'reason' in response_data:
                    error_msg = f"ESP32 error: {response_data.get('reason', error_msg)}"
            
            logger.error("ESP32 lock control error: %s | response=%s", error_msg, response_data)
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
    except requests.Timeout:
        logger.error("ESP32 lock control timeout")
        return jsonify({
            'success': False,
            'error': 'ESP32 request timed out. Please check if ESP32 is reachable.'
        }), 500
    except requests.ConnectionError:
        logger.error("ESP32 lock control connection error")
        return jsonify({
            'success': False,
            'error': 'Failed to connect to ESP32. Please check if ESP32 is online.'
        }), 500
    except requests.RequestException as e:
        logger.exception("ESP32 lock control request failed: %s", e)
        return jsonify({
            'success': False,
            'error': f'Failed to communicate with ESP32: {str(e)}'
        }), 500
    except Exception as e:
        logger.exception("Unexpected error controlling lock: %s", e)
        return jsonify({
            'success': False,
            'error': 'Unexpected error occurred'
        }), 500


def create_lock_control_app(esp32_ip: str, host: str = '127.0.0.1', port: int = 5000):
    """Create and configure the Flask app for lock control.
    
    Args:
        esp32_ip: ESP32 IP address
        host: Web server host (default: 127.0.0.1)
        port: Web server port (default: 5000)
    
    Returns:
        Flask app instance
    """
    global _esp32_ip
    _esp32_ip = esp32_ip
    return app


def run_lock_control_server(esp32_ip: str, host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
    """Run the lock control web server in a thread.
    
    Args:
        esp32_ip: ESP32 IP address
        host: Web server host (default: 127.0.0.1)
        port: Web server port (default: 5000)
        debug: Enable debug mode (default: False)
    """
    global _esp32_ip
    _esp32_ip = esp32_ip
    
    logger.info("Starting Lock Control Web Server")
    logger.info("ESP32 IP: %s", _esp32_ip)
    logger.info("Server: http://%s:%d", host, port)
    
    # Run Flask in non-debug mode when embedded (to avoid auto-reloader issues)
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def main():
    """Main entry point for standalone web server."""
    global _esp32_ip
    
    parser = argparse.ArgumentParser(description="TLS Lock Control Web Server")
    parser.add_argument(
        '--esp32-ip',
        type=str,
        default='auto',
        help="ESP32 IP address or 'auto' to derive x.y.z.184 (default: auto)"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help="Web server port (default: 5000)"
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help="Web server host (default: 127.0.0.1, use 0.0.0.0 for network access)"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Resolve ESP32 IP
    if args.esp32_ip == 'auto':
        _esp32_ip = _derive_device_ip_last_octet_184()
    else:
        _esp32_ip = args.esp32_ip
    
    run_lock_control_server(_esp32_ip, args.host, args.port, args.debug)


if __name__ == '__main__':
    main()

