"""
ESP32-CAM Helper Module

This module provides utilities to connect to ESP32-CAM devices through:
1. Network discovery and stream connection
2. Serial communication for direct connection
3. Stream management and error handling

Author: AI Assistant
Date: 2024
"""

import cv2
import socket
import threading
import time
import subprocess
import platform
import re
from typing import Optional, List, Tuple, Dict
import requests
from urllib.parse import urlparse

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not available. Serial communication disabled.")


class ESP32CamHelper:
    """Helper class for ESP32-CAM connectivity and management."""
    
    def __init__(self):
        self.current_stream = None
        self.stream_thread = None
        self.is_streaming = False
        self.frame = None
        self.lock = threading.Lock()
        
    def find_esp32_cam_ips(self, network_range: str = None, timeout: float = 1.0, include_serial: bool = True) -> List[str]:
        """
        Find ESP32-CAM devices on the network by scanning for common ESP32-CAM endpoints.
        Also checks for ESP32-CAMs connected via serial that have network connectivity.
        
        Args:
            network_range: Network range to scan (e.g., "192.168.1.0/24")
            timeout: Timeout for each connection attempt
            include_serial: Whether to check serial-connected devices for network IPs
            
        Returns:
            List of IP addresses that respond to ESP32-CAM endpoints
        """
        print("Scanning for ESP32-CAM devices...")
        
        found_ips = []
        
        # Method 1: Network discovery
        network_ips = self._find_network_esp32_cams(network_range, timeout)
        found_ips.extend(network_ips)
        
        # Method 2: Check serial-connected devices for network IPs
        if include_serial and SERIAL_AVAILABLE:
            serial_ips = self._find_serial_esp32_cams_with_network()
            found_ips.extend(serial_ips)
        
        # Remove duplicates while preserving order
        unique_ips = []
        seen = set()
        for ip in found_ips:
            if ip not in seen:
                unique_ips.append(ip)
                seen.add(ip)
        
        return unique_ips
    
    def _find_network_esp32_cams(self, network_range: str = None, timeout: float = 1.0) -> List[str]:
        """Find ESP32-CAM devices via network scanning."""
        # Get local network range if not provided
        if network_range is None:
            network_range = self._get_local_network_range()
        
        # ESP32-CAM control endpoints (port 80)
        control_endpoints = [
            "/start",       # Enable streaming
            "/stop",        # Disable streaming
            "/message?cmd=start",  # Alternative start command
            "/",            # Root endpoint
        ]
        
        # ESP32-CAM stream endpoints (port 81)
        stream_endpoints = [
            "/stream",      # MJPEG stream
        ]
        
        # ESP32-CAM ports: 80 for control, 81 for stream
        control_ports = [80]
        stream_ports = [81]
        
        found_ips = []
        ip_list = self._generate_ip_list(network_range)
        
        for ip in ip_list:
            # First check control server (port 80)
            for port in control_ports:
                for endpoint in control_endpoints:
                    if self._test_esp32_endpoint(ip, port, endpoint, timeout):
                        found_ips.append(f"{ip}:{port}")
                        print(f"Found ESP32-CAM control server at {ip}:{port}")
                        break
                if f"{ip}:{port}" in found_ips:
                    break
            
            # Then check stream server (port 81) - but only if control server found
            if f"{ip}:80" in found_ips:
                for port in stream_ports:
                    for endpoint in stream_endpoints:
                        if self._test_esp32_endpoint(ip, port, endpoint, timeout):
                            print(f"Found ESP32-CAM stream server at {ip}:{port}")
                            break
        
        return found_ips
    
    def _find_serial_esp32_cams_with_network(self) -> List[str]:
        """Find ESP32-CAMs connected via serial that have network connectivity."""
        print("Checking serial-connected devices for network connectivity...")
        
        found_ips = []
        
        try:
            # Get available serial ports
            available_ports = serial.tools.list_ports.comports()
            
            for port_info in available_ports:
                port_name = port_info.device
                
                # Check if this looks like an ESP32-CAM port
                if any(keyword in port_info.description.lower() for keyword in 
                       ['usb', 'serial', 'ch340', 'cp210', 'ftdi', 'esp32']):
                    
                    print(f"Checking serial port {port_name} for network connectivity...")
                    
                    # Try to get IP address from this serial port
                    ip = self._get_ip_from_serial_port(port_name)
                    if ip:
                        # Test if this IP responds to ESP32-CAM endpoints
                        if self._test_esp32_endpoint(ip, 80, "/start", 2.0):
                            found_ips.append(f"{ip}:80")
                            print(f"Found ESP32-CAM at {ip}:80 (connected via {port_name})")
                        elif self._test_esp32_endpoint(ip, 80, "/", 2.0):
                            found_ips.append(f"{ip}:80")
                            print(f"Found ESP32-CAM at {ip}:80 (connected via {port_name})")
        
        except Exception as e:
            print(f"Error checking serial ports: {e}")
        
        return found_ips
    
    def _get_ip_from_serial_port(self, port_name: str) -> Optional[str]:
        """
        Try to get IP address from a serial port.
        This method attempts various approaches to extract IP from serial output.
        """
        try:
            # Try to connect to serial port and read recent output
            ser = serial.Serial(port_name, 115200, timeout=1.0)
            time.sleep(0.5)  # Wait for any pending output
            
            # Read available data
            output = ""
            start_time = time.time()
            while time.time() - start_time < 3.0:  # Read for up to 3 seconds
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            output += line + "\n"
                    except:
                        pass
                time.sleep(0.1)
            
            ser.close()
            
            # Look for IP address patterns in the output
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            matches = re.findall(ip_pattern, output)
            
            if matches:
                # Return the first IP found
                return matches[0]
            
            # If no IP found in output, try to scan common ESP32-CAM IPs
            # that might be associated with this serial port
            return self._scan_common_esp32_ips()
            
        except Exception as e:
            print(f"Error reading from serial port {port_name}: {e}")
            return None
    
    def _scan_common_esp32_ips(self) -> Optional[str]:
        """Scan common ESP32-CAM IP addresses."""
        # Common ESP32-CAM IP patterns
        common_ips = [
            "192.168.1.100", "192.168.1.101", "192.168.1.102",
            "192.168.4.1",   # ESP32 AP mode default
            "192.168.0.100", "192.168.0.101", "192.168.0.102",
            "10.0.0.100",    "10.0.0.101",    "10.0.0.102"
        ]
        
        for ip in common_ips:
            if self._test_esp32_endpoint(ip, 80, "/start", 1.0):
                return ip
            elif self._test_esp32_endpoint(ip, 80, "/", 1.0):
                return ip
        
        return None
    
    def check_specific_com_port(self, port_name: str = "COM4") -> Optional[str]:
        """
        Check a specific COM port for ESP32-CAM network connectivity.
        
        Args:
            port_name: COM port name to check (default: "COM4")
            
        Returns:
            IP address if found, None otherwise
        """
        print(f"Checking {port_name} for ESP32-CAM network connectivity...")
        
        if not SERIAL_AVAILABLE:
            print("Serial communication not available")
            return None
        
        try:
            # First, try to get IP from serial output
            ip = self._get_ip_from_serial_port(port_name)
            if ip:
                print(f"Found IP {ip} from {port_name} serial output")
                return ip
            
            # If no IP from serial, scan common ESP32-CAM IPs
            print(f"No IP found in {port_name} output, scanning common ESP32-CAM IPs...")
            ip = self._scan_common_esp32_ips()
            if ip:
                print(f"Found ESP32-CAM at {ip} (likely connected via {port_name})")
                return ip
            
            print(f"No ESP32-CAM found for {port_name}")
            return None
            
        except Exception as e:
            print(f"Error checking {port_name}: {e}")
            return None
    
    def test_specific_ip(self, ip: str, control_port: int = 80, stream_port: int = 81) -> bool:
        """
        Test if a specific IP address is an ESP32-CAM.
        
        Args:
            ip: IP address to test
            control_port: Control server port (default: 80)
            stream_port: Stream server port (default: 81)
            
        Returns:
            True if ESP32-CAM found at this IP, False otherwise
        """
        print(f"Testing IP {ip} for ESP32-CAM...")
        
        # Test control server endpoints (port 80)
        control_endpoints = ["/start", "/stop", "/flash", "/message?cmd=start", "/"]
        
        for endpoint in control_endpoints:
            if self._test_esp32_endpoint(ip, control_port, endpoint, 3.0):
                print(f"✓ ESP32-CAM control server found at {ip}:{control_port} (endpoint: {endpoint})")
                return True
        
        # Test stream server (port 81)
        if self._test_esp32_endpoint(ip, stream_port, "/stream", 3.0):
            print(f"✓ ESP32-CAM stream server found at {ip}:{stream_port}")
            return True
        
        print(f"✗ No ESP32-CAM found at {ip}")
        print(f"  Tried control endpoints on port {control_port}: {control_endpoints}")
        print(f"  Tried stream endpoint on port {stream_port}: /stream")
        return False
    
    def _get_local_network_range(self) -> str:
        """Get the local network range based on the current machine's IP."""
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Extract network prefix (assumes /24 subnet)
            network_prefix = ".".join(local_ip.split(".")[:-1])
            return f"{network_prefix}.0/24"
        except Exception as e:
            print(f"Error getting local network: {e}")
            return "192.168.1.0/24"  # Default fallback
    
    def _generate_ip_list(self, network_range: str) -> List[str]:
        """Generate list of IP addresses from network range."""
        if "/24" in network_range:
            base_ip = network_range.replace("/24", "")
            base_parts = base_ip.split(".")
            if len(base_parts) == 3:
                return [f"{base_parts[0]}.{base_parts[1]}.{base_parts[2]}.{i}" for i in range(1, 255)]
        return []
    
    def _test_esp32_endpoint(self, ip: str, port: int, endpoint: str, timeout: float) -> bool:
        """Test if an endpoint responds (likely ESP32-CAM)."""
        try:
            url = f"http://{ip}:{port}{endpoint}"
            print(f"  Testing: {url}")
            response = requests.get(url, timeout=timeout, stream=True)
            
            print(f"  Response: {response.status_code}")
            
            # For stream endpoints, check for MJPEG content type or 403 (not started)
            if "/stream" in endpoint:
                content_type = response.headers.get('content-type', '').lower()
                print(f"  Content-Type: {content_type}")
                if 'multipart' in content_type or 'mjpeg' in content_type:
                    return True
                # Stream returns 403 Forbidden until /start is called
                if response.status_code == 403:
                    print(f"  Stream endpoint found (403 - needs /start)")
                    return True
            
            # For control endpoints (/start, /stop, /flash, /message), check for successful response
            if response.status_code == 200:
                print(f"  Control endpoint found (200 OK)")
                return True
            
            # Also accept other success status codes
            if 200 <= response.status_code < 300:
                print(f"  Endpoint found ({response.status_code})")
                return True
                    
        except Exception as e:
            print(f"  Error: {e}")
        return False
    
    def connect_to_stream(self, ip: str, control_port: int = 80, stream_port: int = 81) -> bool:
        """
        Connect to ESP32-CAM video stream following proper usage flow.
        
        Args:
            ip: IP address of ESP32-CAM
            control_port: Control server port (default: 80)
            stream_port: Stream server port (default: 81)
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to ESP32-CAM at {ip}...")
            
            # Step 1: Enable streaming via control server
            if not self._enable_streaming(ip, control_port):
                print(f"Failed to enable streaming on control server {ip}:{control_port}")
                return False
            
            # Step 2: Connect to stream server
            stream_url = f"http://{ip}:{stream_port}/stream"
            print(f"Connecting to ESP32-CAM stream: {stream_url}")
            
            # Start streaming thread
            self.is_streaming = True
            self.stream_thread = threading.Thread(
                target=self._stream_worker, 
                args=(stream_url,),
                daemon=True
            )
            self.stream_thread.start()
            
            # Wait a moment for stream to initialize
            time.sleep(2)
            
            if self.current_stream is not None:
                print(f"Successfully connected to ESP32-CAM at {ip}")
                return True
            else:
                print(f"Failed to initialize stream from {ip}")
                return False
                
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            return False
    
    def _enable_streaming(self, ip: str, port: int = 80) -> bool:
        """
        Enable streaming on ESP32-CAM control server.
        
        Args:
            ip: IP address of ESP32-CAM
            port: Control server port (default: 80)
            
        Returns:
            True if streaming enabled successfully, False otherwise
        """
        try:
            # Try /start endpoint first
            start_url = f"http://{ip}:{port}/start"
            response = requests.get(start_url, timeout=5.0)
            
            if response.status_code == 200:
                print(f"Streaming enabled via /start endpoint")
                return True
            
            # Try alternative /message endpoint
            message_url = f"http://{ip}:{port}/message?cmd=start"
            response = requests.get(message_url, timeout=5.0)
            
            if response.status_code == 200:
                print(f"Streaming enabled via /message endpoint")
                return True
            
            print(f"Failed to enable streaming. Status codes: /start={response.status_code}")
            return False
            
        except Exception as e:
            print(f"Error enabling streaming: {e}")
            return False
    
    def _disable_streaming(self, ip: str, port: int = 80) -> bool:
        """
        Disable streaming on ESP32-CAM control server.
        
        Args:
            ip: IP address of ESP32-CAM
            port: Control server port (default: 80)
            
        Returns:
            True if streaming disabled successfully, False otherwise
        """
        try:
            # Try /stop endpoint first
            stop_url = f"http://{ip}:{port}/stop"
            response = requests.get(stop_url, timeout=5.0)
            
            if response.status_code == 200:
                print(f"Streaming disabled via /stop endpoint")
                return True
            
            # Try alternative /message endpoint
            message_url = f"http://{ip}:{port}/message?cmd=stop"
            response = requests.get(message_url, timeout=5.0)
            
            if response.status_code == 200:
                print(f"Streaming disabled via /message endpoint")
                return True
            
            print(f"Failed to disable streaming. Status code: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"Error disabling streaming: {e}")
            return False
    
    def _stream_worker(self, stream_url: str):
        """Worker thread for handling video stream."""
        try:
            self.current_stream = cv2.VideoCapture(stream_url)
            
            if not self.current_stream.isOpened():
                print(f"Failed to open stream: {stream_url}")
                return
            
            while self.is_streaming:
                ret, frame = self.current_stream.read()
                if ret:
                    with self.lock:
                        self.frame = frame.copy()
                else:
                    print("Failed to read frame from stream")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Stream worker error: {e}")
        finally:
            if self.current_stream:
                self.current_stream.release()
                self.current_stream = None
    
    def get_frame(self) -> Optional[cv2.Mat]:
        """Get the latest frame from the stream."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def disconnect(self, ip: str = None, control_port: int = 80):
        """
        Disconnect from the current stream and optionally disable streaming.
        
        Args:
            ip: IP address of ESP32-CAM (optional, for disabling streaming)
            control_port: Control server port (default: 80)
        """
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=5)
        
        if self.current_stream:
            self.current_stream.release()
            self.current_stream = None
        
        with self.lock:
            self.frame = None
        
        # Optionally disable streaming on the ESP32-CAM
        if ip is not None:
            self._disable_streaming(ip, control_port)
        
        print("Disconnected from ESP32-CAM stream")
    
    def set_flash(self, ip: str, on: bool = True, intensity: int = 200, control_port: int = 80) -> bool:
        """
        Control ESP32-CAM flash.
        
        Args:
            ip: IP address of ESP32-CAM
            on: Whether to turn flash on or off
            intensity: Flash intensity (0-255, only used when on=True)
            control_port: Control server port (default: 80)
            
        Returns:
            True if flash command sent successfully, False otherwise
        """
        try:
            if on:
                # Flash on with intensity
                flash_url = f"http://{ip}:{control_port}/flash?on=1&intensity={intensity}"
                response = requests.get(flash_url, timeout=5.0)
                
                if response.status_code == 200:
                    print(f"Flash turned on with intensity {intensity}")
                    return True
                
                # Try alternative /message endpoint
                message_url = f"http://{ip}:{control_port}/message?cmd=flash_on&intensity={intensity}"
                response = requests.get(message_url, timeout=5.0)
                
                if response.status_code == 200:
                    print(f"Flash turned on with intensity {intensity} (via /message)")
                    return True
            else:
                # Flash off
                flash_url = f"http://{ip}:{control_port}/flash?on=0"
                response = requests.get(flash_url, timeout=5.0)
                
                if response.status_code == 200:
                    print("Flash turned off")
                    return True
                
                # Try alternative /message endpoint
                message_url = f"http://{ip}:{control_port}/message?cmd=flash_off"
                response = requests.get(message_url, timeout=5.0)
                
                if response.status_code == 200:
                    print("Flash turned off (via /message)")
                    return True
            
            print(f"Failed to control flash. Status code: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"Error controlling flash: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if currently connected to a stream."""
        return self.is_streaming and self.current_stream is not None
    
    def get_stream_info(self) -> Dict:
        """Get information about the current stream."""
        if not self.is_connected():
            return {}
        
        try:
            width = int(self.current_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.current_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.current_stream.get(cv2.CAP_PROP_FPS)
            
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "is_connected": True
            }
        except Exception as e:
            print(f"Error getting stream info: {e}")
            return {"is_connected": False}


class ESP32CamSerialHelper:
    """Helper class for ESP32-CAM serial communication."""
    
    def __init__(self):
        self.serial_connection = None
        self.is_connected = False
    
    def find_esp32_ports(self) -> List[str]:
        """
        Find available COM ports that might be ESP32-CAM devices.
        
        Returns:
            List of COM port names
        """
        if not SERIAL_AVAILABLE:
            print("Serial communication not available. Install pyserial: pip install pyserial")
            return []
        
        ports = []
        available_ports = serial.tools.list_ports.comports()
        
        for port in available_ports:
            # Common ESP32-CAM identifiers
            if any(keyword in port.description.lower() for keyword in 
                   ['usb', 'serial', 'ch340', 'cp210', 'ftdi', 'esp32']):
                ports.append(port.device)
                print(f"Found potential ESP32-CAM port: {port.device} - {port.description}")
        
        return ports
    
    def connect_serial(self, port: str, baudrate: int = 115200, timeout: float = 1.0) -> bool:
        """
        Connect to ESP32-CAM via serial port.
        
        Args:
            port: COM port name (e.g., "COM3" on Windows, "/dev/ttyUSB0" on Linux)
            baudrate: Serial baudrate (default: 115200)
            timeout: Serial timeout
            
        Returns:
            True if connection successful, False otherwise
        """
        if not SERIAL_AVAILABLE:
            print("Serial communication not available")
            return False
        
        try:
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            
            # Test connection
            time.sleep(2)  # Wait for ESP32 to initialize
            self.serial_connection.write(b'\n')  # Send newline to trigger response
            
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8', errors='ignore')
                print(f"ESP32-CAM response: {response.strip()}")
                self.is_connected = True
                return True
            else:
                print("No response from ESP32-CAM")
                return False
                
        except Exception as e:
            print(f"Error connecting to serial port {port}: {e}")
            return False
    
    def send_command(self, command: str) -> str:
        """
        Send command to ESP32-CAM via serial.
        
        Args:
            command: Command string to send
            
        Returns:
            Response from ESP32-CAM
        """
        if not self.is_connected or not self.serial_connection:
            return ""
        
        try:
            # Send command
            self.serial_connection.write(f"{command}\n".encode('utf-8'))
            time.sleep(0.1)
            
            # Read response
            response = ""
            while self.serial_connection.in_waiting > 0:
                response += self.serial_connection.readline().decode('utf-8', errors='ignore')
            
            return response.strip()
            
        except Exception as e:
            print(f"Error sending command: {e}")
            return ""
    
    def get_ip_address(self) -> Optional[str]:
        """
        Get IP address from ESP32-CAM via serial.
        
        Returns:
            IP address string or None if not found
        """
        # Common commands to get IP address
        commands = ["ip", "wifi", "status", "info"]
        
        for cmd in commands:
            response = self.send_command(cmd)
            if response:
                # Look for IP address pattern
                ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
                matches = re.findall(ip_pattern, response)
                if matches:
                    return matches[0]
        
        return None
    
    def disconnect_serial(self):
        """Disconnect from serial port."""
        if self.serial_connection:
            self.serial_connection.close()
            self.serial_connection = None
            self.is_connected = False
            print("Disconnected from ESP32-CAM serial port")


def main():
    """Example usage of ESP32-CAM helper functions."""
    print("ESP32-CAM Helper - Example Usage")
    print("=" * 40)
    
    helper = ESP32CamHelper()
    
    # Method 1: Check specific COM port (like COM4)
    print("\n1. Checking COM4 for ESP32-CAM:")
    com4_ip = helper.check_specific_com_port("COM4")
    if com4_ip:
        print(f"Found ESP32-CAM at {com4_ip} (connected via COM4)")
        
        # Test the connection
        if helper.test_specific_ip(com4_ip):
            print(f"✓ Confirmed ESP32-CAM at {com4_ip}")
            
            # Try to connect
            print(f"\n2. Connecting to {com4_ip}")
            if helper.connect_to_stream(com4_ip, 80, 81):
                print("Stream connected successfully!")
                
                # Display stream info
                info = helper.get_stream_info()
                if info:
                    print(f"Stream info: {info}")
                
                # Simulate some frame captures
                print("\n3. Capturing frames (press Ctrl+C to stop):")
                try:
                    for i in range(5):
                        frame = helper.get_frame()
                        if frame is not None:
                            print(f"Captured frame {i+1}: {frame.shape}")
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping...")
                
                helper.disconnect(com4_ip, 80)
        else:
            print(f"✗ Could not confirm ESP32-CAM at {com4_ip}")
    else:
        print("No ESP32-CAM found on COM4")
    
    # Method 2: General network discovery
    print("\n4. General Network Discovery:")
    found_ips = helper.find_esp32_cam_ips()
    
    if found_ips:
        print(f"Found {len(found_ips)} ESP32-CAM devices:")
        for ip in found_ips:
            print(f"  - {ip}")
    else:
        print("No ESP32-CAM devices found on network")
    
    # Method 3: Test specific IP if you know it
    print("\n5. Test Specific IP (if you know the IP):")
    known_ips = ["192.168.1.100", "192.168.4.1", "192.168.0.100"]
    for test_ip in known_ips:
        if helper.test_specific_ip(test_ip):
            print(f"Found ESP32-CAM at {test_ip}!")
            break
    else:
        print("No ESP32-CAM found at common IP addresses")
    
    # Serial communication example
    print("\n4. Serial Communication:")
    serial_helper = ESP32CamSerialHelper()
    ports = serial_helper.find_esp32_ports()
    
    if ports:
        print(f"Found {len(ports)} potential ESP32-CAM ports:")
        for port in ports:
            print(f"  - {port}")
        
        # Try to connect to first port
        first_port = ports[0]
        print(f"\n5. Connecting to serial port {first_port}")
        if serial_helper.connect_serial(first_port):
            print("Serial connection successful!")
            
            # Try to get IP address
            ip = serial_helper.get_ip_address()
            if ip:
                print(f"ESP32-CAM IP address: {ip}")
            else:
                print("Could not retrieve IP address")
            
            serial_helper.disconnect_serial()
    else:
        print("No ESP32-CAM serial ports found")


if __name__ == "__main__":
    main()
