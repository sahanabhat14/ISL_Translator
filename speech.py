"""Offline speech on its own thread so camera processing stays responsive."""
import queue
import threading
import time


class SpeechGate:
    def __init__(self):
        self.last_label = None
        self.last_time = float('-inf')
        self.absent_since = None

    def observe_hands(self, present, now):
        if present:
            self.absent_since = None
        elif self.absent_since is None:
            self.absent_since = now
        elif now - self.absent_since >= 1.0:
            self.last_label = None

    def should_speak(self, label, now):
        if label == self.last_label or now - self.last_time < 2.0:
            return False
        self.last_label, self.last_time = label, now
        return True

    def reset(self):
        self.last_label = None


class Speaker:
    def __init__(self):
        self.pending = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.completed = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        engine = None
        initialized = False
        try:
            import pythoncom
            pythoncom.CoInitialize()
            initialized = True
            from win32com.client import Dispatch
            engine = Dispatch('SAPI.SpVoice')
            engine.Rate = 0
            engine.Volume = 100
            self.ready.set()
            while not self.stopping.is_set():
                try:
                    text, created = self.pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                if time.monotonic() - created > 1.0 or self.stopping.is_set():
                    continue
                print(f"Speaking: {text}", flush=True)
                # Synchronous SAPI call on this worker, not on the camera thread.
                # Avoid pyttsx3's repeated runAndWait cancellation on Windows.
                engine.Speak(text, 0)
                self.completed += 1
        except Exception as error:
            self.error = str(error)
            print(f"Speech unavailable: {error}. Text recognition remains available.", flush=True)
        finally:
            self.ready.set()
            engine = None
            if initialized:
                pythoncom.CoUninitialize()

    def say(self, text):
        if self.error or not self.ready.is_set():
            return
        self.clear()
        try:
            self.pending.put_nowait((text, time.monotonic()))
        except queue.Full:
            pass

    def clear(self):
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass

    def close(self):
        self.stopping.set()
        self.clear()
        self.thread.join(timeout=2)
