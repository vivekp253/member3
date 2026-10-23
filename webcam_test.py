
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    print("Check camera permissions or whether another app is using it.")
    raise SystemExit(1)

print("Webcam started! Press Q to close the window.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a webcam frame.")
            break

        cv2.imshow("AI Exam Proctoring - Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
