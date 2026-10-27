"""Keep TensorFlow state outside Streamlit's Keras cleanup on reruns."""
import atexit
import multiprocessing as mp
import threading
from types import SimpleNamespace

import numpy as np


def serve_model(connection, model_dir):
    detector = None
    try:
        import tensorflow as tf
        from recognize import load_run
        from pathlib import Path
        model, report = load_run(Path(model_dir))
        predict = tf.function(lambda batch: model(batch, training=False),
                              input_signature=[tf.TensorSpec([None, 30, 126], tf.float32)],
                              autograph=False)
        predict(np.zeros((1, 30, 126), dtype=np.float32))
        import mediapipe as mediapipe
        options = mediapipe.tasks.vision.HandLandmarkerOptions(
            base_options=mediapipe.tasks.BaseOptions(model_asset_path=str(Path(model_dir).parent / "hand_landmarker.task")),
            running_mode=mediapipe.tasks.vision.RunningMode.VIDEO, num_hands=2)
        detector = mediapipe.tasks.vision.HandLandmarker.create_from_options(options)
        connection.send(("ready", report))
        while True:
            message = connection.recv()
            if message is None:
                break
            if message[0] == "predict":
                connection.send(("prediction", np.asarray(predict(message[1]))))
            elif message[0] == "track":
                result = detector.detect_for_video(mediapipe.Image(
                    image_format=mediapipe.ImageFormat.SRGB, data=message[1]), message[2])
                hands = [[(lm.x, lm.y, lm.z) for lm in hand] for hand in result.hand_landmarks]
                handedness = [entry[0].category_name for entry in result.handedness]
                connection.send(("landmarks", (hands, handedness)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send(("error", str(error)))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if detector is not None:
            detector.close()
        connection.close()


class ModelProcess:
    def __init__(self, model_dir):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.lock = threading.Lock()
        self.process = context.Process(target=serve_model, args=(child, str(model_dir)),
                                       daemon=True, name="isl-inference")
        self.process.start()
        child.close()
        atexit.register(self.close)
        try:
            kind, value = self._receive(90)
            if kind != "ready":
                raise RuntimeError(f"Model could not start: {value}")
            self.report = value
        except Exception:
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise RuntimeError("Model process did not respond. Stop and restart the app.")
        try:
            return self.connection.recv()
        except EOFError as error:
            raise RuntimeError("Model process stopped. Restart the app.") from error

    def __call__(self, batch, training=False):
        if training:
            raise ValueError("The app predictor is for inference only")
        return self._request(("predict", np.asarray(batch, dtype=np.float32)), "prediction")

    def _request(self, message, expected):
        with self.lock:
            try:
                if not self.process.is_alive():
                    raise RuntimeError("Model process stopped. Restart the app.")
                self.connection.send(message)
                kind, value = self._receive(10)
                if kind != expected:
                    raise RuntimeError(f"Prediction failed: {value}")
                return value
            except Exception:
                # Do not reuse a pipe with an old response after a timeout.
                self.close()
                raise

    def hand_tracker(self):
        return RemoteHandTracker(self)

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.connection.close()


class RemoteHandTracker:
    """Shared tracking resources with increasing timestamps across restarts."""
    def __init__(self, model):
        self.model = model

    def detect_for_video(self, rgb, timestamp):
        import time
        # MediaPipe VIDEO timestamps must increase across camera restarts too.
        timestamp = int(time.monotonic() * 1000)
        timestamp = max(timestamp, getattr(self.model, "last_tracking_timestamp", -1) + 1)
        self.model.last_tracking_timestamp = timestamp
        hands, labels = self.model._request(("track", np.ascontiguousarray(rgb), timestamp), "landmarks")
        return SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in hand] for hand in hands],
            handedness=[[SimpleNamespace(category_name=label)] for label in labels])

    def close(self):
        # Camera Stop preserves the cached process for the next Start.
        pass
