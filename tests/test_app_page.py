"""Exercise Streamlit controls without opening a camera or playing speech."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class FakeCameraSession:
    def __init__(self, *args, **kwargs):
        self.alive = False
        self.thread = SimpleNamespace(is_alive=lambda: self.alive)
        self.reset_event = threading.Event()
        self.test_audio = threading.Event()
        self.speech_enabled = kwargs.get("speech", True)

    def start(self):
        self.alive = True

    def stop(self):
        self.alive = False

    def set_speech(self, enabled):
        self.speech_enabled = enabled

    def snapshot(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8) if self.alive else None
        return dict(frame=frame, status="Show your hands", error=None,
                    hands=0, speech="On" if self.speech_enabled else "Muted", history=[])


@unittest.skipUnless(importlib.util.find_spec("streamlit") and
                     (ROOT / "models/hand_landmarker.task").exists(),
                     "Needs app dependencies and local model assets")
class AppPageTests(unittest.TestCase):
    def test_start_stop_reset_audio_and_speech_controls(self):
        from streamlit.testing.v1 import AppTest
        with patch("app_runtime.CameraSession", FakeCameraSession):
            page = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            self.assertEqual(len(page.exception), 0)
            self.assertEqual(page.title[0].value, "ISL Translator")

            def button(label):
                return next(b for b in page.button if b.label == label)

            button("Start camera").click().run()
            self.assertEqual(len(page.exception), 0)
            session = page.session_state.camera_session
            self.assertTrue(session.alive)
            button("Start camera").click().run()
            self.assertIs(page.session_state.camera_session, session)
            page.toggle[0].set_value(False).run()
            self.assertFalse(session.speech_enabled)
            button("Clear / reset").click().run()
            self.assertTrue(session.reset_event.is_set())
            button("Test audio").click().run()
            self.assertTrue(session.test_audio.is_set())
            button("Stop").click().run()
            self.assertFalse(session.alive)
            self.assertEqual(len(page.exception), 0)
            self.assertEqual(len(page.expander), 1)


if __name__ == "__main__":
    unittest.main()
