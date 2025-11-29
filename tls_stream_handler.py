import asyncio
import logging
import time
from typing import Optional, Any

import cv2
import requests


class StreamWatcher:
    def __init__(self, ip: str = "192.168.254.168", session: Optional[Any] = None) -> None:
        self.ip = ip
        self.base_url = f"http://{self.ip}:81"
        self.stream_url = f"{self.base_url}/stream"
        self.logger = logging.getLogger("stream_watcher")
        self._last_online: Optional[bool] = None
        self._stop = asyncio.Event()
        self.main_stream_active: Optional[dict] = None  # Shared flag from main app

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def check_stream_status(self) -> Optional[dict]:
        """Check ESP32 status endpoint instead of directly checking stream.
        
        Returns:
            Status dict if successful, None if failed
        """
        try:
            status_url = f"http://{self.ip}:80/status"
            resp = requests.get(status_url, timeout=(1.5, 2.5))
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    self.logger.debug("Status check successful | streaming=%s | irGate=%s | awaitingVerification=%s",
                                    data.get("streaming"), data.get("irGate"), data.get("awaitingVerification"))
                    return data
                except Exception as exc:
                    self.logger.debug("Failed to parse status JSON: %s", exc)
                    return None
            else:
                self.logger.debug("Status check returned non-200 status: %s", resp.status_code)
                return None
        except requests.RequestException as exc:
            self.logger.debug("Status check request failed: %s", exc)
            return None
        except Exception as exc:
            self.logger.exception("Unexpected error during status check: %s", exc)
            return None

    async def monitor(self) -> None:
        self.logger.info("Starting ESP32-CAM watcher | ip=%s", self.ip)
        try:
            while not self._stop.is_set():
                # Skip checks if main app has stream active to avoid conflicts
                if self.main_stream_active is not None and self.main_stream_active.get("value", False):
                    # Main stream is active, reduce check frequency and don't spam
                    self.logger.debug("Main stream active - skipping watcher check")
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                
                status = await self.check_stream_status()

                # Print status based on status endpoint
                if status is not None:
                    streaming = status.get("streaming", False)
                    ir_gate = status.get("irGate", False)
                    awaiting = status.get("awaitingVerification", False)
                    online = streaming and ir_gate and awaiting
                    
                    if online:
                        print("[OK] Stream Ready — %s" % self.ip)
                        self.logger.info("Stream Ready — %s", self.ip)
                    else:
                        print("[WARN] Stream Not Ready — %s" % self.ip)
                        self.logger.warning("Stream Not Ready — streaming=%s irGate=%s awaiting=%s", 
                                          streaming, ir_gate, awaiting)
                else:
                    print("[WARN] Status Check Failed — %s" % self.ip)
                    self.logger.warning("Status Check Failed")

                # Note: We no longer open stream directly - main app handles that based on status
                current_online = status is not None and status.get("streaming", False) if status else False
                self._last_online = current_online
                
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received. Shutting down watcher...")
        finally:
            self.logger.info("Watcher stopped.")

    async def open_stream(self, duration: int = 10) -> None:
        cap = None
        start = time.perf_counter()
        try:
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                self.logger.warning("Failed to open stream at %s", self.stream_url)
                print("[WARN] Could not open stream. Returning to standby.")
                return

            print("Stream active...")
            self.logger.info("Stream active...")
            while (time.perf_counter() - start) < max(1, int(duration)):
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.logger.warning("Stream read failed (possibly dropped). Closing early.")
                    print("[WARN] Stream dropped — Closing and returning to standby.")
                    break
                # Optional processing could go here (e.g., cv2.imshow or OpenCV analysis)
                await asyncio.sleep(0)  # yield control to event loop
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            print("Stream closed — Returning to standby.")
            self.logger.info("Stream closed — Returning to standby.")



