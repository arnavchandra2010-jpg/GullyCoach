
import cv2
import mediapipe as mp
import time
import math
import csv
from pathlib import Path

folder = Path(__file__).resolve().parent
model_path = folder / "pose_landmarker_lite.task"
output_path = folder / "gullycoach_measurements.csv"

alpha = 0.3
visibility_threshold = 0.6
max_missing_frames = 2

connections = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29),
    (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30),
    (30, 32), (28, 32)
]

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

def measure_knee(landmarks, indices, width, height):
    points = [landmarks[i] for i in indices]

    if any(
        p.visibility < visibility_threshold
        for p in points
    ):
        return None

    return calculate_angle(
        *points,
        width,
        height
    )

def measure_stance_width(landmarks, width, height):
    indices = (23, 24, 25, 26, 27, 28)

    if any(
        landmarks[i].visibility < visibility_threshold
        for i in indices
    ):
        return None

    def point(index):
        landmark = landmarks[index]
        return (
            landmark.x * width,
            landmark.y * height
        )

    left_hip = point(23)
    right_hip = point(24)

    left_knee = point(25)
    right_knee = point(26)

    left_ankle = point(27)
    right_ankle = point(28)

    ankle_distance = math.dist(
        left_ankle,
        right_ankle
    )

    left_leg_length = (
        math.dist(left_hip, left_knee)
        + math.dist(left_knee, left_ankle)
    )

    right_leg_length = (
        math.dist(right_hip, right_knee)
        + math.dist(right_knee, right_ankle)
    )

    average_leg_length = (
        left_leg_length + right_leg_length
    ) / 2

    if average_leg_length < 1e-6:
        return None

    return ankle_distance / average_leg_length

def smooth_angle(previous, current):
    if current is None:
        return None

    if previous is None:
        return current

    return alpha * current + (1 - alpha) * previous

def draw_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

if not model_path.is_file():
    raise FileNotFoundError(
        f"MediaPipe model not found: {model_path}"
    )

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot access webcam.")

previous_smoothed = {
    "left": None,
    "right": None
}

missing_frames = {
    "left": 0,
    "right": 0
}

start_time = time.monotonic()
last_timestamp = -1

try:
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "time_seconds",
            "left_raw",
            "left_smoothed",
            "right_raw",
            "right_smoothed",
            "stance_ratio"
        ])

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as detector:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Failed to capture frame.")
                    break

                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                elapsed = time.monotonic() - start_time

                timestamp = max(
                    last_timestamp + 1,
                    int(elapsed * 1000)
                )

                last_timestamp = timestamp

                result = detector.detect_for_video(
                    image,
                    timestamp
                )

                raw_angles = {
                    "left": None,
                    "right": None
                }

                stance_ratio = None
                ankle_points = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    raw_angles["left"] = measure_knee(
                        landmarks,
                        (23, 25, 27),
                        width,
                        height
                    )

                    raw_angles["right"] = measure_knee(
                        landmarks,
                        (24, 26, 28),
                        width,
                        height
                    )

                    stance_ratio = measure_stance_width(
                        landmarks,
                        width,
                        height
                    )

                    if stance_ratio is not None:
                        left_ankle = landmarks[27]
                        right_ankle = landmarks[28]

                        ankle_points = (
                            (
                                int(left_ankle.x * width),
                                int(left_ankle.y * height)
                            ),
                            (
                                int(right_ankle.x * width),
                                int(right_ankle.y * height)
                            )
                        )

                    for a, b in connections:
                        first = landmarks[a]
                        second = landmarks[b]

                        if (
                            first.visibility >= visibility_threshold
                            and second.visibility >= visibility_threshold
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
                                frame,
                                p1,
                                p2,
                                (0, 255, 0),
                                2
                            )

                    for point in landmarks:
                        if point.visibility >= visibility_threshold:
                            x = int(point.x * width)
                            y = int(point.y * height)

                            cv2.circle(
                                frame,
                                (x, y),
                                5,
                                (0, 0, 255),
                                -1
                            )

                filtered_angles = {}

                for side in ("left", "right"):
                    raw = raw_angles[side]

                    if raw is None:
                        missing_frames[side] += 1

                        if missing_frames[side] > max_missing_frames:
                            previous_smoothed[side] = None

                        filtered_angles[side] = None

                    else:
                        if missing_frames[side] > max_missing_frames:
                            previous_smoothed[side] = None

                        filtered = smooth_angle(
                            previous_smoothed[side],
                            raw
                        )

                        previous_smoothed[side] = filtered
                        missing_frames[side] = 0
                        filtered_angles[side] = filtered

                writer.writerow([
                    round(elapsed, 3),
                    raw_angles["left"],
                    filtered_angles["left"],
                    raw_angles["right"],
                    filtered_angles["right"],
                    stance_ratio
                ])

                if ankle_points is not None:
                    cv2.line(
                        frame,
                        ankle_points[0],
                        ankle_points[1],
                        (0, 255, 255),
                        3
                    )

                    for point in ankle_points:
                        cv2.circle(
                            frame,
                            point,
                            9,
                            (255, 255, 0),
                            -1
                        )

                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (5, 5),
                    (min(width - 5, 545), 200),
                    (30, 30, 30),
                    -1
                )

                cv2.addWeighted(
                    overlay,
                    0.65,
                    frame,
                    0.35,
                    0,
                    frame
                )

                draw_text(
                    frame,
                    "GullyCoach V0.1.5",
                    (20, 40),
                    (0, 255, 255)
                )

                for side, y in [
                    ("left", 85),
                    ("right", 120)
                ]:
                    raw = raw_angles[side]
                    filtered = filtered_angles[side]

                    if raw is not None and filtered is not None:
                        knee_text = (
                            f"{side.title()} knee: "
                            f"{raw:.1f} | Smooth: {filtered:.1f}"
                        )
                    else:
                        knee_text = (
                            f"{side.title()} knee: Not visible"
                        )

                    draw_text(
                        frame,
                        knee_text,
                        (20, y)
                    )

                if stance_ratio is not None:
                    stance_text = (
                        f"Stance width ratio: {stance_ratio:.2f}"
                    )

                    stance_color = (0, 255, 0)

                else:
                    stance_text = "Stance width: Not visible"
                    stance_color = (0, 165, 255)

                draw_text(
                    frame,
                    stance_text,
                    (20, 160),
                    stance_color
                )

                draw_text(
                    frame,
                    "Press Q to exit",
                    (20, 190)
                )

                cv2.imshow(
                    "GullyCoach Batting Analysis",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("GullyCoach session completed.")
print("Measurements saved to:", output_path)
