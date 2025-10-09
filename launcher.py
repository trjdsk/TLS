from __future__ import annotations

import os
import subprocess
import sys


def launch_window() -> int:
    # Launch desktop pipeline
    return subprocess.call([sys.executable, "main.py"])  # inherits console


def launch_web(port: int = 8000, with_discovery: bool = False) -> int:
    env = os.environ.copy()
    if with_discovery:
        # Run discovery broadcaster in background
        subprocess.Popen([sys.executable, "discovery.py"], env=env)
    return subprocess.call([sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(port)])


def main() -> None:
    print("Touchless Lock System Launcher")
    print("1) Launch Window App")
    print("2) Launch Web Server")
    print("3) Launch Web Server + Discovery")
    print("4) Launch Web Server + Discovery + Open UI")
    try:
        choice = input("Select option [1-4]: ").strip()
    except KeyboardInterrupt:
        return
    if choice == "1":
        sys.exit(launch_window())
    elif choice == "2":
        sys.exit(launch_web(with_discovery=False))
    elif choice == "3":
        sys.exit(launch_web(with_discovery=True))
    elif choice == "4":
        # Launch with discovery and open browser
        import webbrowser
        import time
        print("Starting server with discovery...")
        if launch_web(with_discovery=True) == 0:
            print("Opening web UI in browser...")
            time.sleep(2)  # Give server time to start
            webbrowser.open("http://localhost:8000/ui")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()


