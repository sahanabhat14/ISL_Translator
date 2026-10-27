"""Resource ownership and silent-background checks without opening a webcam."""
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from app_runtime import CameraSession, load_model_future


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        load_model_future.cache_clear()

    def test_preview_and_stop_work_while_model_is_loading(self):
        self.run_fake_camera(opened=True, delay_load=True)

    def test_preview_keeps_updating_during_slow_prediction(self):
        self.run_fake_camera(opened=True, delay_prediction=True)

    def test_busy_camera_does_not_release_another_sessions_lock(self):
        lock = threading.Lock()
        lock.acquire()
        session = CameraSession(lock, None)
        session.start()
        session.thread.join(2)
        self.assertFalse(session.thread.is_alive())
        self.assertIn("Another app session", session.snapshot()["error"])
        self.assertTrue(lock.locked())
        lock.release()

    def test_no_sign_is_silent_and_stop_releases_resources(self):
        self.run_fake_camera(opened=True)

    def test_brief_tracking_gaps_do_not_restart_the_full_sequence(self):
        self.run_fake_camera(opened=True, brief_gaps=True)

    def test_failed_camera_releases_detector_and_lock(self):
        self.run_fake_camera(opened=False)

    def test_cleanup_failure_does_not_leave_camera_locked(self):
        self.run_fake_camera(opened=True, fail_cleanup=True)

    def run_fake_camera(self, opened, fail_cleanup=False, delay_load=False, delay_prediction=False, brief_gaps=False):
        lock = threading.Lock()
        finished = threading.Event()
        landmark = NS(x=.2, y=.3, z=.1)
        result = NS(hand_landmarks=[[landmark] * 21],
                    handedness=[[NS(category_name="Left")]])
        def close_detector():
            finished.set()
            if fail_cleanup:
                raise RuntimeError("simulated close failure")
        tracked_frames = [0]
        def detect(*args):
            tracked_frames[0] += 1
            if brief_gaps and tracked_frames[0] % 10 == 0:
                return NS(hand_landmarks=[], handedness=[])
            return result
        detector = NS(detect_for_video=detect, close=close_detector)
        released = threading.Event()
        camera = NS(isOpened=lambda: opened,
                    read=lambda: (True, np.zeros((100, 100, 3), dtype=np.uint8)),
                    release=lambda: released.set())
        calls = []
        speaker_closed = threading.Event()
        ready = threading.Event()
        ready.set()
        speaker = NS(ready=ready, error=None, clear=lambda: None,
                     say=lambda word: calls.append(word), close=lambda: speaker_closed.set())
        loading = threading.Event()
        predicting = threading.Event()
        continue_load = threading.Event()
        continue_prediction = threading.Event()
        def model(*args, **kwargs):
            predicting.set()
            if delay_prediction:
                continue_prediction.wait(10)
            return np.array([[0., 0., 0., 0., 0., 1.]])
        labels = ["HELLO", "THANK_YOU", "YES", "NO", "BYE", "NO_SIGN"]
        def load(*args):
            loading.set()
            if delay_load:
                continue_load.wait(10)
            return model, {"labels": labels}
        fake_cv = NS(VideoCapture=lambda n: camera, flip=lambda f, n: f,
                     cvtColor=lambda f, n: f, COLOR_BGR2RGB=1, circle=lambda *a: None)
        fake_mp = NS(Image=lambda **kw: kw, ImageFormat=NS(SRGB=1),
                     tasks=NS(BaseOptions=lambda **kw: kw, vision=NS(
                         HandLandmarkerOptions=lambda **kw: kw, RunningMode=NS(VIDEO=1),
                         HandLandmarker=NS(create_from_options=lambda opts: detector))))
        with patch.dict("sys.modules", {"cv2": fake_cv, "mediapipe": fake_mp}), \
             patch("app_runtime.load_warmed_model", side_effect=load), \
             patch("app_runtime.create_hand_tracker", return_value=detector), \
             patch("app_runtime.Speaker", return_value=speaker):
            session = CameraSession(lock, None)
            session.start()
            deadline = time.monotonic() + 6
            try:
                if delay_load or delay_prediction:
                    self.assertTrue((loading if delay_load else predicting).wait(5), session.snapshot()["error"])
                    count = session.snapshot()["captured"]
                    time.sleep(.25)
                    self.assertGreater(session.snapshot()["captured"], count)
                    if delay_load:
                        self.assertIn("Loading", session.snapshot()["status"])
                        before = time.monotonic()
                        session.stop()
                        self.assertLess(time.monotonic() - before, 1.0)
                    else:
                        session.stop()
                    self.assertTrue(released.is_set())
                    continue_load.set()
                    continue_prediction.set()
                    session.thread.join(2)
                    self.assertFalse(session.thread.is_alive())
                    self.assertFalse(lock.locked())
                    return
                while session.thread.is_alive() and time.monotonic() < deadline:
                    if session.snapshot()["status"] == "No sign":
                        break
                    time.sleep(.05)
                if opened:
                    self.assertEqual(session.snapshot()["status"], "No sign", session.snapshot()["error"])
                    self.assertEqual(calls, [])
                else:
                    self.assertIn("Camera unavailable", session.snapshot()["error"])
            finally:
                continue_load.set()
                continue_prediction.set()
                session.stop()
            self.assertFalse(session.thread.is_alive())
            self.assertTrue(released.is_set())
            self.assertEqual(finished.is_set(), opened)
            self.assertFalse(lock.locked())
            if opened:
                self.assertTrue(speaker_closed.is_set())


if __name__ == "__main__":
    unittest.main()
