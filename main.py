import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

start = time.monotonic()

with HandLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start) * 1000)
        result = detector.detect_for_video(image, timestamp)

        h, w, _ = frame.shape

        for hand in result.hand_landmarks:
            points = []

            coordinates = []

            for lm in hand:
                 coordinates.extend([lm.x, lm.y, lm.z])

            print("Hand coordinates:", len(coordinates))

            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for a, b in connections:
                cv2.line(frame, points[a], points[b],
                         (255, 0, 0), 2)

        cv2.imshow("ISL Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
