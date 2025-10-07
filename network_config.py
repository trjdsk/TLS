"""
Network Configuration Helper for Touchless Lock System

This script helps configure network access and DNS for the web server.
"""

import socket
import subprocess
import sys
import platform
from typing import List, Tuple, Optional

def get_local_ip_addresses() -> List[Tuple[str, str]]:
    """Get all local IP addresses."""
    addresses = []
    
    try:
        # Get hostname
        hostname = socket.gethostname()
        
        # Get all IP addresses
        ip_list = socket.gethostbyname_ex(hostname)[2]
        
        for ip in ip_list:
            if not ip.startswith("127."):  # Skip localhost
                addresses.append((ip, "Local Network"))
        
        # Try to get external IP (if connected to internet)
        try:
            import requests
            response = requests.get("https://api.ipify.org", timeout=5)
            if response.status_code == 200:
                external_ip = response.text.strip()
                addresses.append((external_ip, "External (Internet)"))
        except:
            pass
            
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
    
    return addresses

def get_network_info():
    """Display network information."""
    print("🌐 Network Configuration Information")
    print("=" * 50)
    
    # Get local IPs
    addresses = get_local_ip_addresses()
    
    if addresses:
        print("📡 Available IP Addresses:")
        for ip, description in addresses:
            print(f"   {ip} - {description}")
    else:
        print("❌ No network interfaces found")
        return
    
    # Get system info
    print(f"\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Hostname: {socket.gethostname()}")
    
    # Port suggestions
    print(f"\n🔌 Recommended Ports:")
    print(f"   Default: 8000")
    print(f"   Alternative: 8080, 3000, 5000")
    print(f"   HTTPS: 8443, 443")

def setup_hosts_file(dns_name: str, ip_address: str):
    """Setup hosts file entry for custom DNS."""
    system = platform.system().lower()
    
    if system == "windows":
        hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
    else:
        hosts_path = "/etc/hosts"
    
    print(f"🔧 Setting up DNS entry: {dns_name} -> {ip_address}")
    print(f"📝 Hosts file location: {hosts_path}")
    print(f"📋 Add this line to your hosts file:")
    print(f"   {ip_address}    {dns_name}")
    print(f"\n⚠️  Note: You may need administrator/root privileges to edit the hosts file")

def check_firewall_windows(port: int):
    """Check Windows firewall configuration."""
    try:
        print(f"🔥 Checking Windows Firewall for port {port}...")
        
        # Check if port is open
        result = subprocess.run([
            "netsh", "advfirewall", "firewall", "show", "rule", 
            f"name=all", "dir=in", "protocol=tcp", f"localport={port}"
        ], capture_output=True, text=True)
        
        if "No rules match" in result.stdout:
            print(f"⚠️  Port {port} is not open in Windows Firewall")
            print(f"🔧 To open the port, run as Administrator:")
            print(f"   netsh advfirewall firewall add rule name=\"TLS Web Server\" dir=in action=allow protocol=TCP localport={port}")
        else:
            print(f"✅ Port {port} appears to be open in Windows Firewall")
            
    except Exception as e:
        print(f"❌ Error checking firewall: {e}")

def check_firewall_linux(port: int):
    """Check Linux firewall configuration."""
    try:
        print(f"🔥 Checking Linux Firewall for port {port}...")
        
        # Check if ufw is active
        result = subprocess.run(["ufw", "status"], capture_output=True, text=True)
        if "Status: active" in result.stdout:
            print(f"⚠️  UFW firewall is active. You may need to allow port {port}:")
            print(f"   sudo ufw allow {port}")
        else:
            print(f"✅ UFW firewall is not active")
            
    except Exception as e:
        print(f"❌ Error checking firewall: {e}")

def check_firewall(port: int):
    """Check firewall configuration based on OS."""
    system = platform.system().lower()
    
    if system == "windows":
        check_firewall_windows(port)
    elif system == "linux":
        check_firewall_linux(port)
    else:
        print(f"🔧 Please check your firewall settings for port {port}")

def generate_startup_commands():
    """Generate startup commands for different scenarios."""
    addresses = get_local_ip_addresses()
    
    if not addresses:
        print("❌ No network interfaces available")
        return
    
    primary_ip = addresses[0][0]
    
    print(f"\n🚀 Startup Commands:")
    print("=" * 50)
    
    # Local mode
    print("🏠 Local Mode (localhost only):")
    print(f"   python launcher.py web")
    print(f"   Access: http://localhost:8000")
    
    # Network mode
    print(f"\n🌐 Network Mode (accessible from other devices):")
    print(f"   python launcher.py web --network --host {primary_ip}")
    print(f"   Access: http://{primary_ip}:8000")
    
    # Custom DNS
    print(f"\n🔧 Custom DNS Mode:")
    print(f"   python launcher.py web --network --host {primary_ip} --dns palm-lock.local")
    print(f"   Access: http://palm-lock.local:8000")
    print(f"   (After setting up hosts file)")
    
    # Different ports
    print(f"\n🔌 Alternative Ports:")
    for port in [8080, 3000, 5000]:
        print(f"   python launcher.py web --network --host {primary_ip} --port {port}")
        print(f"   Access: http://{primary_ip}:{port}")

def main():
    """Main configuration helper."""
    print("Touchless Lock System - Network Configuration Helper")
    print("=" * 60)
    
    # Display network info
    get_network_info()
    
    # Check firewall
    check_firewall(8000)
    
    # Generate commands
    generate_startup_commands()
    
    # DNS setup help
    print(f"\n📋 DNS Setup Instructions:")
    print("=" * 50)
    print("1. Choose a DNS name (e.g., 'palm-lock.local')")
    print("2. Get your local IP address from above")
    print("3. Edit your hosts file:")
    print("   Windows: C:\\Windows\\System32\\drivers\\etc\\hosts")
    print("   Linux/Mac: /etc/hosts")
    print("4. Add line: [YOUR_IP]    [DNS_NAME]")
    print("5. Run: python launcher.py web --network --dns [DNS_NAME]")

if __name__ == "__main__":
    main()
