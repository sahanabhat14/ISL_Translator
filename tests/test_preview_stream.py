"""Check real preview delivery, latest-frame behavior and shutdown without a webcam."""
import http.client
import unittest
from urllib.parse import urlsplit

import cv2
import numpy as np

from preview_stream import PreviewStream


class PreviewStreamTests(unittest.TestCase):
    def test_jpeg_stream_delivers_latest_frame_and_closes(self):
        stream = PreviewStream()
        url = urlsplit(stream.url)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
        try:
            original = np.zeros((720, 1280, 3), dtype=np.uint8)
            stream.publish(original)
            stream.publish(np.full_like(original, 255))
            connection.request("GET", url.path)
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
            self.assertEqual(response.readline(), b"--frame\r\n")
            self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
            length = int(response.readline().split(b":")[1])
            response.readline()
            decoded = cv2.imdecode(np.frombuffer(response.read(length), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (360, 640, 3))
            self.assertGreater(decoded.mean(), 250)  # Latest frame, not the queued black frame.
            self.assertEqual(original.shape, (720, 1280, 3))
            self.assertEqual(original.max(), 0)  # Recognition input remains unchanged.
        finally:
            connection.close()
            stream.close()
        self.assertFalse(stream.thread.is_alive())

    def test_other_paths_do_not_expose_preview(self):
        stream = PreviewStream()
        url = urlsplit(stream.url)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
        try:
            connection.request("GET", "/")
            self.assertEqual(connection.getresponse().status, 404)
        finally:
            connection.close()
            stream.close()
