"""Live recognition using the selected model's vocabulary. Q quits; R resets."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from dataset import FRAMES, FEATURES, validate
from hand_features import coordinates
from speech import Speaker, SpeechGate

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "models" / "v1.0.0"


class StablePrediction:
    """Require three consecutive confident windows; clear on uncertainty."""
    def __init__(self, labels, threshold=0.8):
        self.labels = labels
        self.threshold = threshold
        self.votes = deque(maxlen=3)

    def reset(self):
        self.votes.clear()

    def update(self, probabilities):
        probabilities = np.asarray(probabilities)
        if not np.isfinite(probabilities).all():
            self.reset()
            return "Uncertain"
        index = int(np.argmax(probabilities))
        if probabilities[index] < self.threshold:
            self.reset()
            return "Uncertain"
        self.votes.append(index)
        if len(self.votes) == 3 and len(set(self.votes)) == 1:
            if self.labels[index] == "NO_SIGN":
                return "No sign"
            return self.labels[index].replace("_", " ")
        return "Checking..."


def load_run(path):
    import tensorflow as tf
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(path / "isl_lstm.keras", compile=False)
    if report["input_shape"] != [FRAMES, FEATURES] or model.input_shape[1:] != (FRAMES, FEATURES):
        raise ValueError("Model must accept 30 frames of 126 hand coordinates")
    if model.output_shape[-1] != len(report["labels"]):
        raise ValueError("Model output does not match report labels")
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--check", action="store_true", help="Check model and detector without opening a camera")
    parser.add_argument("--mute", action="store_true", help="Start with speech muted")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    import cv2
    import mediapipe as mp
    model, report = load_run(args.model_dir)
    # Warm up inference before capture so its first call does not stall the camera.
    model(np.zeros((1, FRAMES, FEATURES), dtype=np.float32), training=False)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "models/hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        if args.check:
            for filename in report["split"]["test"]:
                sample_path = Path(filename)
                if not sample_path.is_absolute():
                    sample_path = ROOT / sample_path
                if not sample_path.exists():
                    raise FileNotFoundError("Saved evaluation clips are not included in GitHub. Restore the local dataset to run --check.")
                sample = np.load(sample_path, allow_pickle=False)
                validate(sample)
                prediction = np.asarray(model(sample[None], training=False))
                assert prediction.shape == (1, len(report["labels"]))
                assert np.isfinite(prediction).all()
            print("Model, held-out clip inference and hand detector checked. Webcam not opened.")
            return
        cap = cv2.VideoCapture(args.camera)
        buffer = deque(maxlen=FRAMES)
        smoother = StablePrediction(report["labels"], args.threshold)
        status = "Show your hands"
        step, last_timestamp = 0, -1
        start = time.monotonic()
        speaker = Speaker()
        speech_gate = SpeechGate()
        muted = args.mute
        spoken_labels = {label.replace("_", " ") for label in report["labels"] if label != "NO_SIGN"}
        print("Q: quit. R: reset. M: mute/unmute. T: audio test.")
        print("Supported signs:", ", ".join(report["labels"]))
        try:
            if not cap.isOpened():
                raise RuntimeError("Camera unavailable. Close other camera apps or try --camera 1.")
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Could not read camera frame")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(last_timestamp + 1, int((time.monotonic() - start) * 1000))
                last_timestamp = timestamp
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                data = coordinates(result)
                speech_gate.observe_hands(bool(np.any(data)), time.monotonic())
                if not np.any(data):
                    buffer.clear()
                    smoother.reset()
                    step = 0
                    status = "Show your hands"
                else:
                    buffer.append(data)
                    step += 1
                    if len(buffer) < FRAMES:
                        status = f"Collecting {len(buffer)}/{FRAMES} frames"
                    elif step % 5 == 0:
                        probabilities = np.asarray(model(np.asarray(buffer)[None], training=False))[0]
                        status = smoother.update(probabilities)
                        if "NO_SIGN" in report["labels"] and report["labels"][int(np.argmax(probabilities))] == "NO_SIGN":
                            speaker.clear()
                        if status == "No sign":
                            speech_gate.reset()
                        if (not muted and speaker.ready.is_set() and not speaker.error
                                and status in spoken_labels
                                and speech_gate.should_speak(status, time.monotonic())):
                            speaker.say(status.capitalize())
                h, w = frame.shape[:2]
                for hand in result.hand_landmarks:
                    for lm in hand:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 255, 0), -1)
                audio_status = "unavailable" if speaker.error else "muted" if muted else "on" if speaker.ready.is_set() else "starting"
                for row, text in enumerate([status, f"Hands: {len(result.hand_landmarks)} | Q quit | R reset",
                                            f"Speech: {audio_status} | M mute | T audio test"]):
                    cv2.putText(frame, text, (15, 35 + 35*row), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
                cv2.imshow("ISL Recognition", frame)
                # Match collection's 50 ms wait plus processing; inference adds some overhead.
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q") or cv2.getWindowProperty("ISL Recognition", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    speech_gate.reset()
                    speaker.clear()
                    buffer.clear()
                    smoother.reset()
                    step = 0
                    status = "Reset"
                if key == ord("m"):
                    muted = not muted
                    speaker.clear()
                    speech_gate.reset()
                if key == ord("t"):
                    # Explicit test works even while muted or no hands are visible.
                    speaker.say("Audio is working")
        finally:
            speaker.close()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
