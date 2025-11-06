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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def check_stream_status(self) -> bool:
        try:
            # Synchronous request per specification
            resp = requests.get(self.stream_url, timeout=(1.5, 2.5), stream=True)
            status = int(getattr(resp, "status_code", 0) or 0)
            ctype = (resp.headers.get("Content-Type") or "").lower()
            self.logger.debug("Probe %s -> status=%s ctype=%s", self.stream_url, status, ctype)

            # Explicit IR gate closed
            if status == 403:
                return False

            # Some firmwares return HTML when gate is closed
            if "text/html" in ctype:
                return False

            # Treat any 2xx as online (M-JPEG header may be missing)
            if 200 <= status < 300:
                return True

            return False
        except requests.RequestException as exc:
            self.logger.debug("Stream check failed: %s", exc)
            return False
        except Exception as exc:
            self.logger.exception("Unexpected error during stream status check: %s", exc)
            return False

    async def monitor(self) -> None:
        self.logger.info("Starting ESP32-CAM watcher | ip=%s", self.ip)
        try:
            while not self._stop.is_set():
                online = await self.check_stream_status()

                # Print status every 2 seconds
                if online:
                    print("[OK] Stream Online — %s" % self.ip)
                    self.logger.info("Stream Online — %s", self.ip)
                else:
                    print("[WARN] Stream Offline — Retrying...")
                    self.logger.warning("Stream Offline — Retrying...")

                # IR-trigger assumption: stream becomes available when IR is active
                if online and (self._last_online is False or self._last_online is None):
                    self.logger.info("IR Trigger detected — Opening stream for 10 seconds...")
                    print("IR Trigger detected — Opening stream for 10 seconds...")
                    try:
                        await self.open_stream(duration=10)
                    except Exception as exc:
                        self.logger.exception("Error while opening/reading stream: %s", exc)

                self._last_online = online
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



