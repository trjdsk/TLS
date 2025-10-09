"""
UDP discovery broadcaster for the Touchless Lock System server.
Broadcasts JSON with server IP and port on UDP port 8888 at a fixed interval.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Optional
import platform


def _get_host_ip() -> str:
    """Get the actual network IP address that other devices can reach."""
    # Method 1: Connect to external service to get outbound IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127.") and not ip.startswith("::1"):
                return ip
        finally:
            s.close()
    except Exception:
        pass
    
    # Method 2: Try to enumerate network interfaces
    try:
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(['ipconfig'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'IPv4 Address' in line and '192.168.' in line:
                        # Extract IP from next line or current line
                        ip_line = line.split(':')[-1].strip()
                        if not ip_line:
                            continue
                        # Clean up the IP
                        ip = ip_line.split()[0].strip()
                        if ip and not ip.startswith("127."):
                            return ip
        else:
            # Unix/Linux: try to get interface info
            import subprocess
            result = subprocess.run(['ip', 'route', 'get', '8.8.8.8'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'src' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'src' and i + 1 < len(parts):
                                ip = parts[i + 1]
                                if not ip.startswith("127."):
                                    return ip
    except Exception:
        pass
    
    # Method 3: Fallback to hostname resolution
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith("127."):
            return local_ip
    except Exception:
        pass
    
    # Method 4: Try to get any non-loopback interface
    try:
        import netifaces
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info.get('addr', '')
                    if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
                        return ip
    except ImportError:
        pass
    except Exception:
        pass
    
    return "127.0.0.1"


async def broadcast(service_port: int = 8000, interval_sec: float = 2.0, stop: Optional[asyncio.Event] = None) -> None:
    ip = _get_host_ip()
    print(f"Discovery: Broadcasting server at {ip}:{service_port}")
    payload = {
        "service": "touchless_lock_system",
        "server_ip": ip,
        "server_port": service_port,
    }
    data = json.dumps(payload).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.2)

    try:
        while True:
            sock.sendto(data, ("255.255.255.255", 8888))
            await asyncio.sleep(interval_sec)
            if stop is not None and stop.is_set():
                break
    finally:
        sock.close()


async def main():
    await broadcast()


if __name__ == "__main__":
    asyncio.run(main())


