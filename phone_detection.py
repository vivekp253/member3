
from event_logger import save_proctoring_event
import argparse
from uuid import UUID

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"
PHONE_CLASS_ID = 67
CONFIDENCE_THRESHOLD = 0.25


def main():
    parser = argparse.ArgumentParser(
        description="Standalone phone detection monitor"
    )
    parser.add_argument(
        "--exam-id",
        required=True,
        help="Active exam session UUID",
    )
    args = parser.parse_args()

    try:
        exam_id = str(UUID(args.exam_id))
    except ValueError:
        parser.error("--exam-id must be a valid UUID")

    print(f"Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    print(f"Phone detection started for exam: {exam_id}")
    print("Press Q to quit.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        raise SystemExit(1)

    phone_warning_shown = False

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[PHONE_CLASS_ID],
                verbose=False,
            )

            phone_count = 0
            highest_confidence = 0.0

            for result in results:
                for box in result.boxes:
                    phone_count += 1

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )
                    confidence = float(box.conf[0])
                    highest_confidence = max(
                        highest_confidence, confidence
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Possible phone: {confidence:.2f}",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )

            if phone_count > 0:
                status = f"Possible phone detected: {phone_count}"
                color = (0, 0, 255)

                # Save one event per continuous detection episode.
                if not phone_warning_shown:
                    try:
                        saved_event = save_proctoring_event(
                            exam_id=exam_id,
                            event_type="phone_detected",
                            severity="warning",
                            details={
                                "message": (
                                    "YOLO detected a possible "
                                    "mobile phone"
                                ),
                                "phone_count": phone_count,
                                "confidence": round(
                                    highest_confidence, 3
                                ),
                                "source": "yolo11n",
                            },
                        )
                        print("Saved event:", saved_event)
                    except Exception as exc:
                        print("Could not save event:", exc)

                    phone_warning_shown = True

            else:
                status = "No phone detected"
                color = (0, 255, 0)
                phone_warning_shown = False

            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            cv2.imshow(
                "AI Exam Proctoring - Phone Detection",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Phone detection closed.")


if __name__ == "__main__":
    main()
