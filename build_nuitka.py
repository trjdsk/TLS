#!/usr/bin/env python3
"""Nuitka build script for Touchless Lock System."""

import subprocess
import sys
import os
from pathlib import Path

def build_nuitka():
    """Build the project with Nuitka."""
    
    # Icon file path
    icon_path = "app_icon.ico"
    
    # Check if icon exists
    if not os.path.exists(icon_path):
        print(f"Warning: Icon file '{icon_path}' not found. Building without icon.")
        icon_option = []
    else:
        icon_option = [f"--windows-icon-from-ico={icon_path}"]
        print(f"Using icon: {icon_path}")
    
    # Main executable
    main_cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--windows-console-mode=attach",  # For Windows console
        "--enable-plugin=tk-inter",  # For manage_users_gui.py
        "--enable-plugin=numpy",  # For numpy support
        "--include-package-data=mediapipe",  # Include MediaPipe data files
        "--include-package-data=opencv-python",  # Include OpenCV data
        "--include-data-dir=utils=utils",  # Include utils directory
        "--include-data-file=palm_auth.db=palm_auth.db",  # Include database if needed
        "--output-dir=dist",
        "--output-filename=TLS.exe",  # Windows executable name
        *icon_option,  # Add icon if available
        "main.py"
    ]
    
    print("Building main application...")
    subprocess.run(main_cmd, check=True)
    
    # User management GUI
    gui_cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--windows-console-mode=disable",  # GUI app, no console
        "--enable-plugin=tk-inter",
        "--output-dir=dist",
        "--output-filename=ManageUsers.exe",
        *icon_option,  # Add icon if available
        "manage_users_gui.py"
    ]
    
    print("Building user management GUI...")
    subprocess.run(gui_cmd, check=True)
    
    # CLI user management
    cli_cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--windows-console-mode=attach",
        "--output-dir=dist",
        "--output-filename=ManageUsersCLI.exe",
        *icon_option,  # Add icon if available
        "manage_users.py"
    ]
    
    print("Building CLI user management...")
    subprocess.run(cli_cmd, check=True)
    
    print("\nBuild complete! Executables are in the 'dist' directory.")

if __name__ == "__main__":
    build_nuitka()






