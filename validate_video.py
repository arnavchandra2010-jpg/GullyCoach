
import cv2
import mediapipe as mp
import math
import csv
from pathlib import Path

VIDEO_PATH = "test_video.mp4"
MODEL_PATH = "pose_landmarker_lite.task"
OUTPUT_PATH = "knee_measurements.csv"

def calculate_angle(a, b, c, width, height):
    ax, ay = a.x * width, a.y * height
    bx, by = b.x * width, b.y * height
    cx, cy = c.x * width, c.y * height

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]

    length_ab = math.hypot(*ab)
    length_cb = math.hypot(*cb)

    if length_ab < 1e-6 or length_cb < 1e-6:
        return None

    cosine = dot / (length_ab * length_cb)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))

def knee_angle(landmarks, indices, width, height):
    points = [landmarks[i] for i in indices]

    if any(p.visibility < 0.6 for p in points):
        return None

    return calculate_angle(*points, width, height)

connections = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 31), (28, 32)
]

for path in [VIDEO_PATH, MODEL_PATH]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")

camera = cv2.VideoCapture(VIDEO_PATH)

if not camera.isOpened():
    raise RuntimeError("Cannot open video file.")

fps = camera.get(cv2.CAP_PROP_FPS)

if not math.isfinite(fps) or fps <= 0:
    camera.release()
    raise RuntimeError("Video FPS is invalid.")

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

frame_number = 0
last_timestamp = -1

try:
    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "time_seconds",
            "left_knee_degrees",
            "right_knee_degrees"
        ])

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as detector:

            while True:
                success, frame = camera.read()

                if not success:
                    break

                height, width = frame.shape[:2]

                timestamp = max(
                    last_timestamp + 1,
                    round(frame_number * 1000 / fps)
                )
                last_timestamp = timestamp

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect_for_video(
                    image, timestamp
                )

                left_knee = None
                right_knee = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    left_knee = knee_angle(
                        landmarks, (23, 25, 27),
                        width, height
                    )

                    right_knee = knee_angle(
                        landmarks, (24, 26, 28),
                        width, height
                    )

                    for a, b in connections:
                        first = landmarks[a]
                        second = landmarks[b]

                        if (
                            first.visibility >= 0.6
                            and second.visibility >= 0.6
                        ):
                            p1 = (
                                int(first.x * width),
                                int(first.y * height)
                            )
                            p2 = (
                                int(second.x * width),
                                int(second.y * height)
                            )

                            cv2.line(
                                frame, p1, p2,
                                (0, 255, 0), 2
                            )

                    for point in landmarks:
                        if point.visibility >= 0.6:
                            x = int(point.x * width)
                            y = int(point.y * height)

                            cv2.circle(
                                frame, (x, y), 4,
                                (0, 0, 255), -1
                            )

                writer.writerow([
                    frame_number,
                    frame_number / fps,
                    round(left_knee, 2)
                    if left_knee is not None else "",
                    round(right_knee, 2)
                    if right_knee is not None else ""
                ])

                left_text = (
                    f"Left knee: {left_knee:.1f}"
                    if left_knee is not None
                    else "Left knee: N/A"
                )

                right_text = (
                    f"Right knee: {right_knee:.1f}"
                    if right_knee is not None
                    else "Right knee: N/A"
                )

                cv2.putText(
                    frame, left_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                cv2.putText(
                    frame, right_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                cv2.imshow(
                    "GullyCoach Video Validation",
                    frame
                )

                frame_number += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Processed {frame_number} frames.")
print(f"Measurements saved to {OUTPUT_PATH}")
