
import cv2
import mediapipe as mp
import math
import csv
from pathlib import Path

folder = Path(__file__).resolve().parent
video_path = folder / "test_video.mp4"
model_path = folder / "pose_landmarker_lite.task"
output_path = folder / "knee_diagnostic.csv"

start_frame = 105
end_frame = 140
visibility_threshold = 0.6
jump_threshold = 0.025

video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise RuntimeError("Cannot open video.")

fps = video.get(cv2.CAP_PROP_FPS)

if not math.isfinite(fps) or fps <= 0:
    video.release()
    raise RuntimeError("Invalid video FPS.")

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1
)

previous_positions = {
    "left": None,
    "right": None
}

last_timestamp = -1
frame_number = 0

try:
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "time_seconds",
            "left_x",
            "left_y",
            "right_x",
            "right_y",
            "left_visibility",
            "right_visibility",
            "left_jump",
            "right_jump",
            "left_suspicious",
            "right_suspicious",
            "possible_swap"
        ])

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as detector:

            while frame_number <= end_frame:
                success, frame = video.read()

                if not success:
                    break

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = max(
                    last_timestamp + 1,
                    round(frame_number * 1000 / fps)
                )
                last_timestamp = timestamp

                result = detector.detect_for_video(
                    image, timestamp
                )

                if frame_number < start_frame:
                    frame_number += 1
                    continue

                positions = {}
                visibility = {}
                jumps = {}
                suspicious = {}

                landmarks = (
                    result.pose_landmarks[0]
                    if result.pose_landmarks
                    else None
                )

                for side, index, color in [
                    ("left", 25, (0, 165, 255)),
                    ("right", 26, (0, 0, 255))
                ]:
                    position = None
                    score = 0.0

                    if landmarks:
                        knee = landmarks[index]
                        score = knee.visibility

                        if score >= visibility_threshold:
                            position = (knee.x, knee.y)

                    previous = previous_positions[side]

                    jump = (
                        math.dist(position, previous)
                        if position is not None
                        and previous is not None
                        else None
                    )

                    flag = (
                        jump is not None
                        and jump > jump_threshold
                    )

                    positions[side] = position
                    visibility[side] = score
                    jumps[side] = jump
                    suspicious[side] = flag

                    if position is not None:
                        x = int(position[0] * width)
                        y = int(position[1] * height)

                        cv2.circle(
                            frame, (x, y), 9, color, -1
                        )

                        cv2.putText(
                            frame, side.upper(),
                            (x + 12, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2
                        )

                old_left = previous_positions["left"]
                old_right = previous_positions["right"]
                new_left = positions["left"]
                new_right = positions["right"]

                possible_swap = False

                if all(
                    p is not None
                    for p in [
                        old_left, old_right,
                        new_left, new_right
                    ]
                ):
                    same_cost = (
                        math.dist(old_left, new_left)
                        + math.dist(old_right, new_right)
                    )

                    swapped_cost = (
                        math.dist(old_left, new_right)
                        + math.dist(old_right, new_left)
                    )

                    possible_swap = (
                        swapped_cost < same_cost * 0.7
                        and same_cost > jump_threshold
                    )

                for side in ["left", "right"]:
                    previous_positions[side] = positions[side]

                writer.writerow([
                    frame_number,
                    round(frame_number / fps, 4),
                    positions["left"][0]
                    if positions["left"] else "",
                    positions["left"][1]
                    if positions["left"] else "",
                    positions["right"][0]
                    if positions["right"] else "",
                    positions["right"][1]
                    if positions["right"] else "",
                    visibility["left"],
                    visibility["right"],
                    jumps["left"],
                    jumps["right"],
                    suspicious["left"],
                    suspicious["right"],
                    possible_swap
                ])

                if possible_swap:
                    status = "POSSIBLE KNEE SWAP"
                    status_color = (0, 0, 255)
                elif any(suspicious.values()):
                    status = "SUSPICIOUS MOVEMENT"
                    status_color = (0, 165, 255)
                else:
                    status = "TRACKING"
                    status_color = (0, 255, 0)

                cv2.rectangle(
                    frame, (0, 0), (width, 85),
                    (35, 35, 35), -1
                )

                cv2.putText(
                    frame,
                    f"Frame {frame_number}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

                cv2.putText(
                    frame, status, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_color, 2
                )

                scale = min(
                    1000 / width,
                    600 / height,
                    1.0
                )

                display = cv2.resize(
                    frame, None,
                    fx=scale,
                    fy=scale
                )

                cv2.imshow(
                    "GullyCoach Knee Diagnostic",
                    display
                )

                if cv2.waitKey(150) & 0xFF == ord("q"):
                    break

                frame_number += 1

finally:
    video.release()
    cv2.destroyAllWindows()

print("Diagnostic completed.")
print("Results saved to:", output_path)
