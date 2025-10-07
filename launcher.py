"""
Launcher script for Touchless Lock System

This script allows you to run either:
1. FastAPI web server (recommended)
2. Desktop application (legacy)

Usage:
    python launcher.py web     # Start web server
    python launcher.py desktop # Start desktop app
    python launcher.py         # Start web server (default)
"""

import sys
import argparse
import logging
import socket
import subprocess
import platform

def _is_private(ip: str) -> bool:
    try:
        parts = [int(p) for p in ip.split('.')]
        if len(parts) != 4:
            return False
        if parts[0] == 10:
            return True
        if parts[0] == 192 and parts[1] == 168:
            return True
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return True
        return False
    except Exception:
        return False

def _score_ip(ip: str) -> int:
    try:
        parts = [int(p) for p in ip.split('.')]
        if parts[0] == 192 and parts[1] == 168:
            return 300
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return 200
        if parts[0] == 10:
            return 100
    except Exception:
        pass
    return 0

def _windows_ips_from_ipconfig() -> list[tuple[str, int, bool]]:
    results: list[tuple[str, int, bool]] = []
    try:
        proc = subprocess.run(["ipconfig"], capture_output=True, text=True, check=False)
        output = proc.stdout.splitlines()
        block: list[str] = []
        def process_block(lines: list[str]):
            if not lines:
                return
            header = " ".join(lines[:2]).lower()
            skip_keywords = ["openvpn", "tap", "tun", "tailscale", "zerotier", "vmware", "virtual", "hyper-v", "ve-", "docker", "loopback"]
            if any(k in header for k in skip_keywords):
                return
            ip: str | None = None
            has_gateway = False
            for ln in lines:
                if "IPv4 Address" in ln or "IPv4-adress" in ln or "IPv4" in ln:
                    if ":" in ln:
                        ip_candidate = ln.split(":", 1)[1].strip().split(" ")[0]
                        if _is_private(ip_candidate):
                            ip = ip_candidate
                if "Default Gateway" in ln and ":" in ln:
                    val = ln.split(":", 1)[1].strip()
                    if val and val != "0.0.0.0":
                        has_gateway = True
            if ip:
                results.append((ip, _score_ip(ip), has_gateway))
        for ln in output:
            if ln.strip() == "":
                process_block(block)
                block = []
            else:
                block.append(ln)
        process_block(block)
    except Exception:
        pass
    return results

def get_local_ip():
    """Get the LAN/WiFi IP address, avoiding VPN/virtual adapters when possible."""
    system = platform.system().lower()
    candidates: list[tuple[str, int, bool]] = []
    if system == "windows":
        candidates = _windows_ips_from_ipconfig()
        if candidates:
            candidates.sort(key=lambda t: (t[2], t[1]), reverse=True)
            return candidates[0][0]
    # Cross-platform fallback
    try:
        hostnames = {socket.gethostname()}
        try:
            hostnames.add(socket.getfqdn())
        except Exception:
            pass
        addrs = set()
        for hn in hostnames:
            try:
                infos = socket.getaddrinfo(hn, None, family=socket.AF_INET)
                for info in infos:
                    ip = info[4][0]
                    if _is_private(ip):
                        addrs.add(ip)
            except Exception:
                continue
        if addrs:
            best = sorted([(ip, _score_ip(ip)) for ip in addrs], key=lambda x: x[1], reverse=True)
            if best:
                return best[0][0]
    except Exception:
        pass
    # UDP fallback
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if _is_private(ip):
            return ip
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return None

def run_web_server(host="0.0.0.0", port=8000, network_mode=False):
    """Run the FastAPI web server."""
    try:
        from web_server import run_server
        print("Starting Touchless Lock System Web Server...")
        
        if network_mode:
            print(f"🌐 Network Mode: Server accessible from other devices")
            print(f"📱 Local access: http://localhost:{port}")
            print(f"🌍 Network access: http://{host}:{port}")
            print(f"📋 Share this URL with other devices: http://{host}:{port}")
        else:
            print(f"🏠 Local Mode: Server only accessible locally")
            print(f"📱 Open your browser and go to: http://localhost:{port}")
        
        print("Press Ctrl+C to stop the server")
        run_server(host=host, port=port, network_mode=network_mode)
    except ImportError as e:
        print(f"Error importing web server: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting web server: {e}")
        sys.exit(1)

def run_desktop_app():
    """Run the desktop application."""
    try:
        from main import main
        print("Starting Touchless Lock System Desktop Application...")
        print("Press 'r' to register, 'v' to verify, 'q' to quit")
        main()
    except ImportError as e:
        print(f"Error importing desktop app: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting desktop app: {e}")
        sys.exit(1)

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Touchless Lock System Launcher")
    parser.add_argument(
        "mode", 
        nargs="?", 
        choices=["web", "desktop"], 
        default="web",
        help="Mode to run: 'web' for FastAPI server (default), 'desktop' for desktop app"
    )
    parser.add_argument(
        "--host", 
        default=None, 
        help="Host for web server (auto-detected if not specified)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port for web server (default: 8000)"
    )
    parser.add_argument(
        "--network", 
        action="store_true", 
        help="Enable network mode - server accessible from other devices"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    if args.mode == "web":
        print("=" * 50)
        print("Touchless Lock System - Web Server Mode")
        print("=" * 50)
        print("This is the recommended mode for the refactored system.")
        print("The web server provides a modern interface for palm detection.")
        print("=" * 50)
        
        # Auto-detect local IP if not specified
        if args.host is None:
            print("🔍 Auto-detecting local IP address...")
            local_ip = get_local_ip()
            if local_ip:
                args.host = local_ip
                print(f"✅ Detected local IP: {local_ip}")
            else:
                args.host = "0.0.0.0"
                print("⚠️  Could not detect local IP, using 0.0.0.0")
        
        # No DNS feature; show clear IP URLs
        print(f"📱 Local access: http://localhost:{args.port}")
        if args.network:
            print(f"🌍 Network access: http://{args.host}:{args.port}")
        
        run_web_server(host=args.host, port=args.port, network_mode=args.network)
    elif args.mode == "desktop":
        print("=" * 50)
        print("Touchless Lock System - Desktop Mode")
        print("=" * 50)
        print("This is the legacy desktop application.")
        print("Note: ESP32 cam functionality has been removed.")
        print("=" * 50)
        run_desktop_app()

if __name__ == "__main__":
    main()
