"""Latest-frame MJPEG preview, served only on loopback with a per-session URL."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import secrets
import threading
import time


class PreviewStream:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.number = 0
        self.closed = False
        self.path = "/" + secrets.token_urlsafe(24)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != owner.path:
                    self.send_error(404)
                    return
                self.connection.settimeout(3)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                previous = -1
                try:
                    while True:
                        with owner.condition:
                            owner.condition.wait_for(
                                lambda: owner.closed or (owner.jpeg is not None and
                                                         owner.number != previous), timeout=1)
                            if owner.closed:
                                return
                            if owner.jpeg is None or owner.number == previous:
                                continue
                            jpeg, previous = owner.jpeg, owner.number
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                except (OSError, TimeoutError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}{self.path}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": .1}, daemon=True,
                                       name="isl-preview")
        self.thread.start()
        self.times = []
        self.fps = 0.

    def publish(self, rgb):
        import cv2
        # Resize only the display copy. Recognition retains native camera frames.
        if rgb.shape[1] > 640:
            rgb = cv2.resize(rgb, (640, round(rgb.shape[0] * 640 / rgb.shape[1])),
                             interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError("Could not encode camera preview")
        now = time.monotonic()
        self.times = (self.times + [now])[-60:]
        self.fps = ((len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-6)
                    if len(self.times) > 1 else 0.)
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.number += 1
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
