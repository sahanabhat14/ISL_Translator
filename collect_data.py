
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from uuid import uuid4
from dataset import SIGNS, FRAMES, validate, load_dataset
from hand_features import coordinates

ROOT = Path(__file__).resolve().parent

signs = SIGNS
recordings = 30
frames = FRAMES

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = Options(
    base_options=BaseOptions(
        model_asset_path=str(ROOT / "models/hand_landmarker.task")
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not detected")

start = time.monotonic()
_, existing_labels, _, _ = load_dataset(ROOT / "data")

with HandLandmarker.create_from_options(options) as detector:

    def get_coordinates():
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Cannot read camera")

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start) * 1000)
        result = detector.detect_for_video(image, timestamp)

        data = coordinates(result)

        detected = len(result.hand_landmarks)
        cv2.putText(frame, f"Hands detected: {detected}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if detected else (0, 0, 255), 2)
        return frame, data

    try:
        for label_index, sign in enumerate(signs):
            (ROOT / "data" / sign).mkdir(parents=True, exist_ok=True)

            n = int((existing_labels == label_index).sum())
            print(f"{sign}: {n} valid recordings, {max(0, recordings - n)} remaining")
            if sign == "NO_SIGN":
                print("Keep hands visible. Relax them or make ordinary non-sign movements.")
                print("Vary poses across clips; do not perform any of the five trained signs.")
            while n < recordings:
                print(f"Recording {sign}: {n + 1}/{recordings}")

                for count in range(3, 0, -1):
                    frame, data = get_coordinates()

                    cv2.putText(
                        frame, f"{sign} - Get ready: {count}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )

                    cv2.imshow("Collect ISL Data", frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                sequence = []

                for f in range(frames):
                    frame, data = get_coordinates()
                    sequence.append(data)

                    cv2.putText(
                        frame, f"{sign} - {f + 1}/{frames}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )
                    if sign == "NO_SIGN":
                        cv2.putText(frame, "Rest or move naturally; keep hands visible",
                                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 255), 2)

                    cv2.imshow("Collect ISL Data", frame)

                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                sequence = np.asarray(sequence, dtype=np.float32)
                try:
                    validate(sequence)
                except ValueError as error:
                    print(f"Not saved: {error}. Check lighting and keep hands in view.")
                    continue
                destination = ROOT / "data" / sign / f"{uuid4().hex}.npy"
                np.save(destination, sequence)
                n += 1
                print(f"Saved {destination}")

    except KeyboardInterrupt:
        print("Collection stopped")

    finally:
        cap.release()
        cv2.destroyAllWindows()
