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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imshow("Anh goc", frame)
        cv2.imshow("Anh xam", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
             break
        windows = ("Anh goc", "Anh xam")
        closed = any(
            cv2.getWindowProperty(
                name, cv2.WND_PROP_VISIBLE
            ) < 1 for name in windows
        )
        if closed:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
