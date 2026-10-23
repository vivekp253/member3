
import cv2
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
from event_logger import save_proctoring_event
import os
import time
import requests

# =========================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================

load_dotenv()

EXAM_ID = input(
    "Paste the exam_id returned by POST /start-exam: "
).strip()

if not EXAM_ID:
    raise RuntimeError("An active exam_id is required.")

BASE_DIR = Path(__file__).parent

FACE_DETECTOR_MODEL = (
    BASE_DIR / "models" / "blaze_face_short_range.tflite"
)
FACE_LANDMARKER_MODEL = (
    BASE_DIR / "models" / "face_landmarker.task"
)
PHONE_MODEL = BASE_DIR / "yolo11n.pt"

NO_FACE_LIMIT = 15.0
GAZE_AWAY_SECONDS = 3.0
EYE_CLOSED_SECONDS = 0.6

PHONE_CLASS_ID = 67
PHONE_CONFIDENCE = 0.25

HORIZONTAL_THRESHOLD = 0.055
VERTICAL_THRESHOLD = 0.10
DEBUG_INTERVAL = 2.0

API_BASE_URL = "http://127.0.0.1:5000"

for model_path in (
    FACE_DETECTOR_MODEL,
    FACE_LANDMARKER_MODEL,
    PHONE_MODEL,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

print("Model files verified.")
print(f"Active EXAM_ID: {EXAM_ID}")

# =========================================
# 2. EYE-TRACKING FUNCTIONS
# =========================================

EYES = [
    {
        "corners": (33, 133),
        "lids": (159, 145),
        "iris": (469, 470, 471, 472),
    },
    {
        "corners": (362, 263),
        "lids": (386, 374),
        "iris": (474, 475, 476, 477),
    },
]


def measure_eyes(landmarks):
    horizontal = []
    vertical = []
    openness = []

    for eye in EYES:
        corner_a = landmarks[eye["corners"][0]]
        corner_b = landmarks[eye["corners"][1]]
        upper = landmarks[eye["lids"][0]]
        lower = landmarks[eye["lids"][1]]

        iris_points = [
            landmarks[i] for i in eye["iris"]
        ]

        iris_x = sum(p.x for p in iris_points) / len(iris_points)
        iris_y = sum(p.y for p in iris_points) / len(iris_points)

        eye_left = min(corner_a.x, corner_b.x)
        eye_right = max(corner_a.x, corner_b.x)
        eye_top = min(upper.y, lower.y)
        eye_bottom = max(upper.y, lower.y)

        eye_width = eye_right - eye_left
        eye_height = eye_bottom - eye_top

        if eye_width < 0.005 or eye_height < 0.002:
            continue

        horizontal.append(
            (iris_x - eye_left) / eye_width
        )
        vertical.append(
            (iris_y - eye_top) / eye_height
        )
        openness.append(eye_height / eye_width)

    if len(horizontal) != 2:
        return None

    return {
        "x": sum(horizontal) / 2,
        "y": sum(vertical) / 2,
        "eye_openness": openness,
    }


def classify_gaze(x, y, baseline_x, baseline_y):
    dx = x - baseline_x
    dy = y - baseline_y

    horizontal = "FORWARD"
    vertical = ""

    if dx < -HORIZONTAL_THRESHOLD:
        horizontal = "LEFT"
    elif dx > HORIZONTAL_THRESHOLD:
        horizontal = "RIGHT"

    if dy < -VERTICAL_THRESHOLD:
        vertical = "UP"
    elif dy > VERTICAL_THRESHOLD:
        vertical = "DOWN"

    if vertical and horizontal != "FORWARD":
        return f"{vertical}-{horizontal}"

    return vertical if vertical else horizontal


# =========================================
# 3. SUPABASE EVENT LOGGER
# =========================================

def log_event(event_type, details, duration_ms=None):
    try:
        saved_event = save_proctoring_event(
            exam_id=EXAM_ID,
            event_type=event_type,
            severity="warning",
            duration_ms=duration_ms,
            details={
                **details,
                "source": "integrated_monitor",
            },
        )

        print(
            f"[DB SUCCESS] Saved {event_type}: "
            f"{saved_event}"
        )
        return True

    except Exception as exc:
        print(
            f"[DB ERROR] Could not save {event_type}: {exc}"
        )
        return False


# =========================================
# 4. TERMINATE EXAM THROUGH FLASK
# =========================================

def terminate_exam(reason):
    """Recognize both a new termination and an already-terminated session."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/terminate-exam",
            json={
                "exam_id": EXAM_ID,
                "reason": reason,
            },
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            status = result.get("status")

            # The endpoint may return an explicit flag for a new
            # termination, or status='terminated' for a closed session.
            if (
                result.get("exam_terminated") is True
                or status == "terminated"
            ):
                print(
                    f"[EXAM TERMINATED] Reason: {reason}"
                )
                return True

            print(
                f"[TERMINATION NOT CONFIRMED] {result}"
            )
            return False

        print(
            f"[TERMINATION ERROR] HTTP {response.status_code}: "
            f"{response.text}"
        )
        return False

    except (requests.RequestException, ValueError) as exc:
        print(
            f"[TERMINATION ERROR] Cannot confirm termination: {exc}"
        )
        return False


# =========================================
# 5. CONFIGURE MEDIAPIPE
# =========================================

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_DETECTOR_MODEL)
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.6,
)

landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_LANDMARKER_MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=2,
)

# =========================================
# 6. LOAD PHONE DETECTOR
# =========================================

print("Loading YOLO model...")
phone_model = YOLO(str(PHONE_MODEL))
print("YOLO model loaded.")

# =========================================
# 7. TRACKING STATE
# =========================================

baseline_x = None
baseline_y = None

gaze_history = deque(maxlen=7)

no_face_since = None
gaze_away_since = None
closed_since = None

absence_logged = False
multiple_faces_logged = False
phone_logged = False
gaze_away_logged = False
eyes_closed_logged = False

last_debug_print = 0.0
exam_terminated = False

# =========================================
# 8. START WEBCAM
# =========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW_NAME = "AI Exam Proctoring - Integrated Monitor"

print("Integrated monitor started.")
print("Keep your face visible to test detection.")
print("Click the webcam window and press C to calibrate gaze.")
print("Press Q to quit.")
print("Face absence limit: 15 seconds.")

# =========================================
# 9. INTEGRATED MONITOR LOOP
# =========================================

try:
    with (
        FaceDetector.create_from_options(
            face_detector_options
        ) as face_detector,
        FaceLandmarker.create_from_options(
            landmarker_options
        ) as face_landmarker,
    ):
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            now = time.monotonic()
            timestamp_ms = int(now * 1000)

            # FACE DETECTION
            face_result = face_detector.detect(mp_image)
            face_count = len(face_result.detections)

            # FACE LANDMARKS
            landmarker_result = (
                face_landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )
            )

            landmark_count = len(
                landmarker_result.face_landmarks
            )

            # PHONE DETECTION
            phone_results = phone_model.predict(
                source=frame,
                conf=PHONE_CONFIDENCE,
                classes=[PHONE_CLASS_ID],
                verbose=False,
            )

            phone_count = 0
            highest_phone_confidence = 0.0

            for result in phone_results:
                for box in result.boxes:
                    phone_count += 1

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )
                    confidence = float(box.conf[0])

                    highest_phone_confidence = max(
                        highest_phone_confidence,
                        confidence,
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
                        f"Phone: {confidence:.2f}",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

            # FACE ABSENCE AND AUTOMATIC TERMINATION
            if face_count == 0:
                if no_face_since is None:
                    no_face_since = now
                    print(
                        "[DEBUG] No face detected; timer started."
                    )

                absent_seconds = now - no_face_since

                if absent_seconds >= NO_FACE_LIMIT:
                    if not absence_logged:
                        saved = log_event(
                            "face_absent",
                            {
                                "message": (
                                    "No face detected for "
                                    f"{NO_FACE_LIMIT} seconds"
                                ),
                                "duration_seconds": round(
                                    absent_seconds, 2
                                ),
                                "face_count": face_count,
                            },
                            duration_ms=int(
                                absent_seconds * 1000
                            ),
                        )

                        if saved:
                            absence_logged = True

                    print(
                        "[WARNING] Face absent for "
                        f"{absent_seconds:.1f} seconds."
                    )

                    exam_terminated = terminate_exam(
                        "face_absent_15_seconds"
                    )

                    if exam_terminated:
                        break

                    # Avoid hammering the backend if it is unavailable.
                    time.sleep(1)

            else:
                no_face_since = None
                absence_logged = False

            # MULTIPLE FACES EVENT
            if face_count > 1:
                if not multiple_faces_logged:
                    saved = log_event(
                        "multiple_faces",
                        {
                            "face_count": face_count,
                            "message": "Multiple faces detected",
                        },
                    )

                    if saved:
                        multiple_faces_logged = True
            else:
                multiple_faces_logged = False

            # PHONE EVENT
            if phone_count > 0:
                if not phone_logged:
                    saved = log_event(
                        "phone_detected",
                        {
                            "phone_count": phone_count,
                            "confidence": round(
                                highest_phone_confidence, 3
                            ),
                        },
                    )

                    if saved:
                        phone_logged = True
            else:
                phone_logged = False

            # EYE TRACKING
            gaze_label = "NO FACE LANDMARKS"
            measurements = None

            if landmarker_result.face_landmarks:
                landmarks = landmarker_result.face_landmarks[0]
                measurements = measure_eyes(landmarks)

                if measurements is None:
                    gaze_label = "GAZE UNCERTAIN"
                    gaze_history.clear()
                    gaze_away_since = None
                    closed_since = None
                    gaze_away_logged = False
                    eyes_closed_logged = False

                else:
                    both_closed = all(
                        value < 0.10
                        for value in measurements["eye_openness"]
                    )

                    # EYES CLOSED EVENT
                    if both_closed:
                        if closed_since is None:
                            closed_since = now

                        gaze_history.clear()
                        gaze_away_since = None
                        gaze_away_logged = False

                        closed_seconds = now - closed_since

                        if closed_seconds >= EYE_CLOSED_SECONDS:
                            gaze_label = "EYES CLOSED"

                            if not eyes_closed_logged:
                                saved = log_event(
                                    "eyes_closed",
                                    {
                                        "duration_seconds": round(
                                            closed_seconds, 2
                                        ),
                                    },
                                    duration_ms=int(
                                        closed_seconds * 1000
                                    ),
                                )

                                if saved:
                                    eyes_closed_logged = True
                        else:
                            gaze_label = "BLINKING..."

                    # EYES OPEN / GAZE EVENT
                    else:
                        closed_since = None
                        eyes_closed_logged = False

                        if baseline_x is None:
                            gaze_label = (
                                "LOOK FORWARD - PRESS C"
                            )
                            gaze_away_since = None
                            gaze_away_logged = False

                        else:
                            gaze = classify_gaze(
                                measurements["x"],
                                measurements["y"],
                                baseline_x,
                                baseline_y,
                            )

                            gaze_history.append(gaze)

                            display_gaze = max(
                                set(gaze_history),
                                key=gaze_history.count,
                            )

                            gaze_label = f"GAZE: {display_gaze}"

                            if display_gaze != "FORWARD":
                                if gaze_away_since is None:
                                    gaze_away_since = now

                                away_seconds = (
                                    now - gaze_away_since
                                )

                                if (
                                    away_seconds
                                    >= GAZE_AWAY_SECONDS
                                    and not gaze_away_logged
                                ):
                                    saved = log_event(
                                        "gaze_away",
                                        {
                                            "gaze_direction": display_gaze,
                                            "duration_seconds": round(
                                                away_seconds, 2
                                            ),
                                        },
                                        duration_ms=int(
                                            away_seconds * 1000
                                        ),
                                    )

                                    if saved:
                                        gaze_away_logged = True
                            else:
                                gaze_away_since = None
                                gaze_away_logged = False

            else:
                gaze_history.clear()
                gaze_away_since = None
                closed_since = None
                gaze_away_logged = False
                eyes_closed_logged = False

            # DEBUG OUTPUT
            if now - last_debug_print >= DEBUG_INTERVAL:
                print(
                    f"[DEBUG] Faces={face_count} | "
                    f"Landmarks={landmark_count} | "
                    f"Phones={phone_count} | "
                    f"Gaze={gaze_label}"
                )
                last_debug_print = now

            # DRAW FACE BOXES
            height, width = frame.shape[:2]

            for detection in face_result.detections:
                bbox = detection.bounding_box

                left = max(0, bbox.origin_x)
                top = max(0, bbox.origin_y)
                right = min(
                    width, bbox.origin_x + bbox.width
                )
                bottom = min(
                    height, bbox.origin_y + bbox.height
                )

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2,
                )

            # STATUS PANEL
            cv2.rectangle(
                frame,
                (10, 10),
                (630, 125),
                (25, 25, 25),
                -1,
            )

            cv2.putText(
                frame,
                f"Faces: {face_count} | Phones: {phone_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                gaze_label,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 200, 255),
                2,
            )

            if face_count == 0 and no_face_since is not None:
                face_status = (
                    f"No face: {now - no_face_since:.1f}s / "
                    f"{NO_FACE_LIMIT:.0f}s"
                )
            else:
                face_status = (
                    "Face detected" if face_count else "Face absent"
                )

            cv2.putText(
                frame,
                face_status,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            # SHOW FRAME BEFORE READING KEYS
            cv2.imshow(WINDOW_NAME, frame)

            # KEYBOARD INPUT
            key = cv2.waitKey(1)

            if key != -1:
                key &= 0xFF
                print(f"[KEYBOARD TEST] Key received: {key}")

            # GAZE CALIBRATION
            if key in (ord("c"), ord("C")):
                if (
                    landmarker_result.face_landmarks
                    and measurements is not None
                    and not all(
                        value < 0.10
                        for value in measurements["eye_openness"]
                    )
                ):
                    baseline_x = measurements["x"]
                    baseline_y = measurements["y"]

                    gaze_history.clear()
                    gaze_away_since = None
                    gaze_away_logged = False

                    gaze_label = "CALIBRATED - LOOK FORWARD"

                    print(
                        "[DEBUG] Gaze calibration complete: "
                        f"x={baseline_x:.3f}, "
                        f"y={baseline_y:.3f}"
                    )
                else:
                    print(
                        "[DEBUG] Calibration skipped: keep your "
                        "face visible and eyes open, then press C again."
                    )

            if key in (ord("q"), ord("Q")):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Integrated monitor closed.")
