import cv2

cap = cv2.VideoCapture(0)
try:
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc camera")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Khong doc duoc anh")
        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        visible = cv2.getWindowProperty(
            "Camera", cv2.WND_PROP_VISIBLE
        )
        if visible < 1:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
