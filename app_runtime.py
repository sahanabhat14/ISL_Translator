"""Camera worker for the local Streamlit app; no UI calls from this thread."""
from collections import deque
from functools import lru_cache
from concurrent.futures import Future
import threading
import time

import numpy as np
from dataset import FRAMES
from hand_features import coordinates
from recognize import StablePrediction
from inference_worker import ModelProcess
from speech import Speaker, SpeechGate


@lru_cache(maxsize=2)
def load_warmed_model(model_dir):
    """Retain the model across camera restarts instead of rebuilding its state."""
    model = ModelProcess(model_dir)
    return model, model.report


@lru_cache(maxsize=2)
def load_model_future(model_dir):
    """Loading can continue without owning or blocking the webcam."""
    future = Future()
    def load():
        try:
            future.set_result(load_warmed_model(model_dir))
        except Exception as error:
            future.set_exception(error)
    threading.Thread(target=load, daemon=True, name="isl-model-loader").start()
    return future


def create_hand_tracker(model):
    return model.hand_tracker()


class CameraSession:
    def __init__(self, camera_lock, model_dir, camera=0, speech=True):
        self.camera_lock = camera_lock
        self.model_dir, self.camera = model_dir, camera
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.test_audio = threading.Event()
        self.muted = threading.Event()
        if not speech:
            self.muted.set()
        self.last_seen = time.monotonic()
        self.state = dict(frame=None, status="Starting…", error=None, hands=0,
                          speech="Off", history=[], frames=0, ready=False, captured=0,
                          camera_fps=0., tracking_fps=0., tracking_ms=0., prediction_ms=0.)
        self.capture_ready = threading.Event()
        self.capture_error = None
        self.capture_thread = None
        self.latest_rgb = None
        self.frame_number = 0
        self.captured_at = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        self.last_seen = time.monotonic()
        with self.lock:
            return {**self.state, "history": list(self.state["history"])}

    def publish(self, **changes):
        with self.lock:
            self.state.update(changes)

    def set_speech(self, enabled):
        if enabled:
            self.muted.clear()
        else:
            self.muted.set()

    def stop(self):
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1)
        self.thread.join(timeout=2)

    def _capture(self):
        """Only this thread opens, reads and releases the webcam."""
        cap = None
        try:
            import cv2
            backends = [None]
            if hasattr(cv2, "CAP_DSHOW"):
                backends.append(cv2.CAP_DSHOW)
            for backend in backends:
                candidate = cv2.VideoCapture(self.camera) if backend is None else cv2.VideoCapture(self.camera, backend)
                ok, frame = candidate.read() if candidate.isOpened() else (False, None)
                if ok:
                    cap = candidate
                    break
                candidate.release()
            if cap is None:
                raise RuntimeError("Camera unavailable. Close recognize.py or other camera apps, then try again.")
            if hasattr(cap, "set") and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture_times = deque(maxlen=60)
            while not self.stop_event.is_set():
                if time.monotonic() - self.last_seen > 30:
                    self.stop_event.set()
                    break
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                now = time.monotonic()
                capture_times.append(now)
                fps = ((len(capture_times) - 1) / max(capture_times[-1] - capture_times[0], 1e-6)
                       if len(capture_times) > 1 else 0.)
                with self.lock:
                    self.latest_rgb = rgb
                    self.frame_number += 1
                    self.captured_at = now
                    points = self.state.get("points", ()) if now - self.state.get("points_at", 0) < .3 else ()
                    count = self.frame_number
                display = rgb.copy()
                height, width = display.shape[:2]
                for x, y in points:
                    cv2.circle(display, (int(x * width), int(y * height)), 3, (0, 220, 140), -1)
                self.publish(frame=display, captured=count, camera_fps=fps)
                self.capture_ready.set()
                # The camera driver already paces reads. Do not add 50 ms per read.
                if self.stop_event.wait(.001):
                    break
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames. Press Stop and try Start again.")
        except Exception as error:
            self.capture_error = str(error)
            self.publish(error=str(error))
        finally:
            if cap is not None:
                cap.release()
            self.capture_ready.set()

    def _run(self):
        detector, speaker = None, None
        acquired = self.camera_lock.acquire(blocking=False)
        try:
            if not acquired:
                raise RuntimeError("Another app session is using the camera. Stop it or close its tab and wait 30 seconds.")
            self.publish(status="Opening camera…")
            self.capture_thread = threading.Thread(target=self._capture, daemon=True, name="isl-camera")
            self.capture_thread.start()
            while not self.capture_ready.wait(.1):
                if self.stop_event.is_set():
                    return
            if self.capture_error:
                raise RuntimeError(self.capture_error)
            self.publish(status="Loading recognition model…")
            future = load_model_future(self.model_dir)
            while not future.done():
                if self.stop_event.wait(.1):
                    return
                if self.capture_error:
                    raise RuntimeError(self.capture_error)
            model, report = future.result()
            labels = report["labels"]
            if self.stop_event.is_set():
                return
            self.publish(status="Starting hand tracking…")
            detector = create_hand_tracker(model)
            speaker = Speaker()
            self.publish(ready=True)
            smoother = StablePrediction(labels)
            gate = SpeechGate()
            buffer = deque(maxlen=FRAMES)
            known = {label.replace("_", " ") for label in labels if label != "NO_SIGN"}
            step, timestamp = 0, -1
            was_muted = self.muted.is_set()
            start = time.monotonic()
            last_frame = -1
            tracking_times = deque(maxlen=60)
            prediction_ms = 0.
            missing_since = None
            status = "Show your hands"
            while not self.stop_event.is_set():
                if time.monotonic() - self.last_seen > 30:
                    self.publish(status="Stopped after the browser disconnected")
                    break
                if self.reset_event.is_set():
                    self.reset_event.clear()
                    buffer.clear()
                    smoother.reset()
                    gate.reset()
                    speaker.clear()
                    step = 0
                    missing_since = None
                    status = "Show your hands"
                    self.publish(history=[])
                muted = self.muted.is_set()
                if muted != was_muted:
                    speaker.clear()
                    gate.reset()
                    was_muted = muted
                if self.capture_error:
                    raise RuntimeError(self.capture_error)
                with self.lock:
                    number, rgb, captured_at = self.frame_number, self.latest_rgb, self.captured_at
                if number == last_frame or rgb is None:
                    self.stop_event.wait(.005)
                    continue
                last_frame = number
                processing_started = time.monotonic()
                timestamp = max(timestamp + 1, int((captured_at - start) * 1000))
                result = detector.detect_for_video(rgb, timestamp)
                tracking_ms = (time.monotonic() - processing_started) * 1000
                tracking_times.append(processing_started)
                tracking_fps = ((len(tracking_times) - 1) / max(tracking_times[-1] - tracking_times[0], 1e-6)
                                if len(tracking_times) > 1 else 0.)
                data = coordinates(result)
                present = bool(np.any(data))
                gate.observe_hands(present, time.monotonic())
                if not present:
                    if missing_since is None:
                        missing_since = captured_at
                    if captured_at - missing_since >= .2:
                        buffer.clear()
                        smoother.reset()
                        step = 0
                    status = "Keep your hands visible" if buffer else "Show your hands"
                else:
                    missing_since = None
                    buffer.append(data)
                    step += 1
                    if len(buffer) < FRAMES:
                        status = f"Collecting movement: {len(buffer)}/{FRAMES} frames"
                    elif step % 5 == 0:
                        prediction_started = time.monotonic()
                        probabilities = np.asarray(model(np.asarray(buffer)[None], training=False))[0]
                        prediction_ms = (time.monotonic() - prediction_started) * 1000
                        status = smoother.update(probabilities)
                        if labels[int(np.argmax(probabilities))] == "NO_SIGN":
                            speaker.clear()
                        if status == "No sign":
                            gate.reset()
                        if status in known and gate.should_speak(status, time.monotonic()):
                            with self.lock:
                                self.state["history"] = (self.state["history"] + [status.capitalize()])[-30:]
                            if not self.muted.is_set() and speaker.ready.is_set() and not speaker.error:
                                speaker.say(status.capitalize())
                if self.test_audio.is_set():
                    self.test_audio.clear()
                    if speaker.ready.is_set() and not speaker.error:
                        speaker.say("Audio is working")
                audio = (f"Unavailable: {speaker.error}" if speaker.error else
                         "Muted" if self.muted.is_set() else "On" if speaker.ready.is_set() else "Starting…")
                self.publish(status=status, hands=len(result.hand_landmarks),
                             frames=len(buffer), speech=audio,
                             tracking_fps=tracking_fps, tracking_ms=tracking_ms,
                             prediction_ms=prediction_ms,
                             points=[(lm.x, lm.y) for hand in result.hand_landmarks for lm in hand],
                             points_at=captured_at)
                # Match collect_data.py: tracking work followed by a 50 ms wait.
                # Changing temporal sampling changes what the trained LSTM sees.
                # Preview capture stays independent and continues at camera speed.
                self.stop_event.wait(.05)
        except Exception as error:
            self.publish(error=str(error), status="Could not continue")
        finally:
            self.stop_event.set()
            if self.capture_thread is not None:
                self.capture_thread.join(timeout=2)
            for resource, method in [(detector, "close"), (speaker, "close")]:
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception as error:
                        self.publish(error=f"Could not finish camera cleanup: {error}")
            if acquired:
                if self.capture_thread is not None and self.capture_thread.is_alive():
                    self.publish(error="The camera driver is not responding. Close other camera apps before retrying.")
                    def release_when_closed():
                        self.capture_thread.join()
                        self.camera_lock.release()
                    threading.Thread(target=release_when_closed, daemon=True).start()
                else:
                    self.camera_lock.release()
            self.publish(frame=None, hands=0, frames=0, ready=False, speech="Off")
            if not self.state["error"]:
                self.publish(status="Stopped")
