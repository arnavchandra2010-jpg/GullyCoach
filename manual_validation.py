
import cv2
import csv
import math
from pathlib import Path

folder = Path(__file__).resolve().parent
video = cv2.VideoCapture(str(folder / "test_video.mp4"))

frames = [39, 63, 87, 121, 207]
measurements = []
points = []

def calculate_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    lengths = math.hypot(*ab) * math.hypot(*cb)

    if lengths == 0:
        return None

    cosine = max(-1, min(1, dot / lengths))
    return math.degrees(math.acos(cosine))

def mouse_callback(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
        points.append((x, y))

try:
    cv2.namedWindow("Manual Validation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Validation", mouse_callback)

    for frame_number in frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = video.read()

        if not success:
            print(f"Could not read frame {frame_number}")
            continue

        points.clear()

        while True:
            display = image.copy()

            for index, point in enumerate(points):
                cv2.circle(display, point, 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    ["Hip", "Knee", "Ankle"][index],
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

            cv2.putText(
                display,
                f"Frame {frame_number} - Click HIP, KNEE, ANKLE",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

            if len(points) == 3:
                cv2.line(display, points[0], points[1], (0, 255, 0), 2)
                cv2.line(display, points[1], points[2], (0, 255, 0), 2)

            cv2.imshow("Manual Validation", display)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                points.clear()

            if key == ord("q"):
                break

            if key == 13 and len(points) == 3:
                angle = calculate_angle(*points)

                measurements.append([
                    frame_number,
                    round(angle, 2)
                ])

                print(f"Frame {frame_number}: {angle:.2f} degrees")
                break

        if key == ord("q"):
            break

    with open(folder / "manual_angles.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "reference_angle"])
        writer.writerows(measurements)

finally:
    video.release()
    cv2.destroyAllWindows()

print("Manual measurements saved.")
