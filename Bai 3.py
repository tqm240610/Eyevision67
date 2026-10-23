"""Tim vung toi ung vien con nguoi. Q: thoat."""
from pathlib import Path
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


def find_pupil(frame, center, radius):
    cx, cy = center
    if radius < 5:
        return None

    h, w = frame.shape[:2]
    x1 = max(0, int(cx - radius))
    y1 = max(0, int(cy - radius))
    x2 = min(w, int(cx + radius) + 1)
    y2 = min(h, int(cy + radius) + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    yy, xx = np.indices(gray.shape)
    distance = (xx - (cx - x1)) ** 2
    distance += (yy - (cy - y1)) ** 2
    mask = distance < (radius * 0.85) ** 2

    pixels = gray[mask]
    if pixels.size < 20:
        return None

    contrast = float(np.percentile(pixels, 85))
    contrast -= float(np.percentile(pixels, 10))
    if contrast < 18:
        return None

    method = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    threshold, _ = cv2.threshold(
        pixels.reshape(-1, 1), 0, 255, method
    )
    binary = ((gray <= threshold) & mask)
    binary = binary.astype(np.uint8) * 255

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, 8
    )

    best = None
    best_score = float("inf")
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3 or area > pixels.size * 0.45:
            continue

        px, py = centroids[i]
        dx = px - (cx - x1)
        dy = py - (cy - y1)
        center_distance = (dx * dx + dy * dy) ** 0.5

        if center_distance > radius * 0.65:
            continue

        score = center_distance + abs(area - pixels.size * 0.12) * 0.02
        if score < best_score:
            best_score = score
            best = (round(px + x1), round(py + y1))

    return best


def find_model():
    candidates = [
        Path("face_landmarker.task"),
        Path(__file__).with_name("face_landmarker.task"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Khong tim thay face_landmarker.task"
    )


def main():
    model_path = find_model()

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    landmarker = vision.FaceLandmarker.create_from_options(options)
    start_time = time.monotonic()
    last_timestamp = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int((time.monotonic() - start_time) * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = landmarker.detect_for_video(image, timestamp)
            view = frame.copy()
            messages = []

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                eyes = [
                    ("LEFT", [469, 470, 471, 472]),
                    ("RIGHT", [474, 475, 476, 477]),
                ]

                for label, ids in eyes:
                    points = np.array(
                        [
                            (landmarks[i].x * 640,
                             landmarks[i].y * 480)
                            for i in ids
                        ],
                        dtype=np.float32,
                    )
                    center, radius = cv2.minEnclosingCircle(points)
                    pupil = find_pupil(frame, center, radius)
                    cx, cy = center
                    drawn = ( round(cx), round(cy))

                    cv2.circle(
                        view, drawn, round(radius),
                        (255, 255, 0), 1
                    )

                    status = label + ": IRIS ONLY"
                    if pupil is not None:
                        px, py = pupil
                        drawn = (639 - px, py)
                        cv2.drawMarker(
                            view, drawn, (0, 0, 255),
                            cv2.MARKER_CROSS, 10, 1
                        )
                        status = label + ": PUPIL CANDIDATE"

                    messages.append(status)

            for row, message in enumerate(messages):
                cv2.putText(
                    view, message, (10, 30 + 30 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1,
                )

            cv2.imshow("Minh - Pupil", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            visible = cv2.getWindowProperty(
                "Minh - Pupil", cv2.WND_PROP_VISIBLE
            )
            if visible < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()