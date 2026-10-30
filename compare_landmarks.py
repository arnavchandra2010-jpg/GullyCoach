
import cv2
import mediapipe as mp
import math
import csv
from pathlib import Path

folder = Path(__file__).resolve().parent
video_path = str(folder / "test_video.mp4")
model_path = str(folder / "pose_landmarker_lite.task")

frames_to_check = [39, 63, 87, 121, 207]
clicks = []

def angle(a, b, c):
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    size = math.hypot(*v1) * math.hypot(*v2)

    if size == 0:
        return None

    cosine = max(-1, min(1, dot / size))
    return math.degrees(math.acos(cosine))


def mouse_callback(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
        original_x = round(x / scale)
        original_y = round(y / scale)

        original_x = max(0, min(original_x, width - 1))
        original_y = max(0, min(original_y, height - 1))

        clicks.append((original_x, original_y))


video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise RuntimeError("Could not open video")

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Invalid video FPS")

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

predictions = {}
last_timestamp = -1

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
    for frame_number in range(max(frames_to_check) + 1):
        success, frame = video.read()

        if not success:
            break

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            last_timestamp + 1,
            round(frame_number * 1000 / fps)
        )
        last_timestamp = timestamp

        result = detector.detect_for_video(image, timestamp)

        if frame_number in frames_to_check:
            predictions[frame_number] = (
                frame.copy(),
                result.pose_landmarks[0]
                if result.pose_landmarks else None
            )

video.release()

rows = []

cv2.namedWindow("GullyCoach Comparison", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("GullyCoach Comparison", mouse_callback)

for frame_number in frames_to_check:
    if frame_number not in predictions:
        continue

    
    original, landmarks = predictions[frame_number]
    height, width = original.shape[:2]

    max_width = 1000
    max_height = 600

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    clicks.clear()


    left_points = None
    right_points = None

    if landmarks:
        left_points = [
            (
                round(landmarks[i].x * width),
                round(landmarks[i].y * height)
            )
            for i in (23, 25, 27)
        ]

        right_points = [
            (
                round(landmarks[i].x * width),
                round(landmarks[i].y * height)
            )
            for i in (24, 26, 28)
        ]

    while True:
        display = original.copy()

        for points, color in [
            (left_points, (0, 165, 255)),
            (right_points, (0, 0, 255))
        ]:
            if points:
                for point in points:
                    cv2.circle(display, point, 7, color, -1)

                cv2.line(display, points[0], points[1], color, 2)
                cv2.line(display, points[1], points[2], color, 2)

        for i, point in enumerate(clicks):
            cv2.circle(display, point, 6, (0, 255, 0), -1)

            
        cv2.rectangle(
            display,
            (0, 0),
            (width, 65),
            (35, 35, 35),
            -1
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Click Hip > Knee > Ankle | Enter: Save | R: Reset",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )


        if len(clicks) == 3:
            cv2.line(display, clicks[0], clicks[1], (0, 255, 0), 2)
            cv2.line(display, clicks[1], clicks[2], (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Frame {frame_number} | HIP KNEE ANKLE | Enter=save R=retry Q=quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        
        display_small = cv2.resize(
            display,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(
            "GullyCoach Comparison",
            display_small
        )


        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            clicks.clear()

        if key == ord("q"):
            break

        if key == 13 and len(clicks) == 3:
            manual_angle = angle(*clicks)

            row = [
                frame_number,
                manual_angle,
                *[value for point in clicks for value in point]
            ]

            for points in (left_points, right_points):
                if points:
                    row.append(angle(*points))
                    row.extend(
                        value for point in points for value in point
                    )
                else:
                    row.extend([""] * 7)

            rows.append(row)

            cv2.imwrite(
                str(folder / f"comparison_{frame_number}.png"),
                display
            )
            break

    if key == ord("q"):
        break

cv2.destroyAllWindows()

with open(folder / "landmark_comparison.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "manual_angle",
        "manual_hip_x", "manual_hip_y",
        "manual_knee_x", "manual_knee_y",
        "manual_ankle_x", "manual_ankle_y",
        "left_angle",
        "left_hip_x", "left_hip_y",
        "left_knee_x", "left_knee_y",
        "left_ankle_x", "left_ankle_y",
        "right_angle",
        "right_hip_x", "right_hip_y",
        "right_knee_x", "right_knee_y",
        "right_ankle_x", "right_ankle_y"
    ])

    writer.writerows(rows)

print("Landmark comparison complete!")
