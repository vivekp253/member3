import cv2
import mediapipe as mp
from pathlib import Path
from collections import deque
import time
import os
from dotenv import load_dotenv
from event_logger import save_proctoring_event

load_dotenv()

EXAM_ID = os.getenv("EXAM_ID")
if not EXAM_ID:
    raise RuntimeError("EXAM_ID is missing from .env")

# =========================================
# Configuration
# =========================================

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Face Landmarker model not found: {MODEL_PATH}")

HORIZONTAL_THRESHOLD = 0.055
VERTICAL_THRESHOLD = 0.10

GAZE_AWAY_SECONDS = 3.0
EYE_CLOSED_SECONDS = 0.6

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

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

# =========================================
# Eye measurements
# =========================================

def measure_eyes(landmarks):
    horizontal = []
    vertical = []
    openness = []

    for eye in EYES:
        corner_a = landmarks[eye["corners"][0]]
        corner_b = landmarks[eye["corners"][1]]
        upper = landmarks[eye["lids"][0]]
        lower = landmarks[eye["lids"][1]]

        iris_points = [landmarks[i] for i in eye["iris"]]

        iris_x = sum(p.x for p in iris_points) / len(iris_points)
        iris_y = sum(p.y for p in iris_points) / len(iris_points)

        eye_left = min(corner_a.x, corner_b.x)
        eye_right = max(corner_a.x, corner_b.x)
        eye_width = eye_right - eye_left

        eye_top = min(upper.y, lower.y)
        eye_bottom = max(upper.y, lower.y)
        eye_height = eye_bottom - eye_top

        if eye_width < 0.005 or eye_height < 0.002:
            continue

        horizontal.append(
            (iris_x - eye_left) / eye_width
        )
        vertical.append(
            (iris_y - eye_top) / eye_height
        )
        openness.append(
            eye_height / eye_width
        )

    if len(horizontal) != 2:
        return None

    return {
        "x": sum(horizontal) / len(horizontal),
        "y": sum(vertical) / len(vertical),
        "openness": sum(openness) / len(openness),
        "eye_openness": openness,
    }


# =========================================
# Gaze classification
# =========================================

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
# Supabase event logging
# =========================================

def log_event(event_type, details):
    try:
        saved = save_proctoring_event(
            exam_id=EXAM_ID,
            event_type=event_type,
            severity="warning",
            details=details,
        )

        print(
            f"Supabase event saved: {event_type} | {saved}"
        )
        return True

    except Exception as exc:
        print(
            f"Could not save {event_type} event: {exc}"
        )
        return False


# =========================================
# Webcam initialization
# =========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

baseline_x = None
baseline_y = None

gaze_history = deque(maxlen=7)

closed_since = None
gaze_away_since = None

eyes_closed_logged = False
gaze_away_logged = False

print("Calibrated eye tracker started.")
print("Look straight at the screen and press C.")
print("Press Q to quit.")

# =========================================
# Main webcam loop
# =========================================

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(time.monotonic() * 1000)

            result = landmarker.detect_for_video(
                image,
                timestamp_ms
            )

            key = cv2.waitKey(1) & 0xFF

            label = ""
            color = (100, 220, 140)
            measurements = None
            now = time.monotonic()

            # =========================================
            # Face detected
            # =========================================

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]
                measurements = measure_eyes(landmarks)

                if measurements is None:
                    label = "GAZE UNCERTAIN"

                    gaze_history.clear()
                    closed_since = None
                    gaze_away_since = None
                    gaze_away_logged = False

                else:
                    both_closed = all(
                        value < 0.10
                        for value in measurements["eye_openness"]
                    )

                    # =====================================
                    # EYES-CLOSED DETECTION
                    # =====================================

                    if both_closed:

                        if closed_since is None:
                            closed_since = now

                        gaze_history.clear()
                        gaze_away_since = None
                        gaze_away_logged = False

                        if now - closed_since >= EYE_CLOSED_SECONDS:

                            label = "EYES CLOSED"
                            color = (80, 80, 255)

                            if not eyes_closed_logged:

                                saved = log_event(
                                    "eyes_closed",
                                    {
                                        "duration_seconds": round(
                                            now - closed_since, 2
                                        )
                                    },
                                )

                                if saved:
                                    eyes_closed_logged = True

                        else:
                            label = "BLINKING..."

                    # =====================================
                    # EYES OPEN
                    # =====================================

                    else:
                        closed_since = None
                        eyes_closed_logged = False

                        # Wait for calibration
                        if baseline_x is None:

                            label = "LOOK FORWARD - PRESS C"
                            color = (0, 210, 255)

                            gaze_history.clear()
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

                            # Smooth gaze predictions
                            display_gaze = max(
                                set(gaze_history),
                                key=gaze_history.count,
                            )

                            label = f"GAZE: {display_gaze}"

                            # =================================
                            # GAZE-AWAY DETECTION
                            # =================================

                            if display_gaze != "FORWARD":

                                if gaze_away_since is None:
                                    gaze_away_since = now

                                away_duration = (
                                    now - gaze_away_since
                                )

                                if (
                                    away_duration >= GAZE_AWAY_SECONDS
                                    and not gaze_away_logged
                                ):

                                    saved = log_event(
                                        "gaze_away",
                                        {
                                            "gaze_direction": display_gaze,
                                            "duration_seconds": round(
                                                away_duration, 2
                                            ),
                                        },
                                    )

                                    if saved:
                                        gaze_away_logged = True

                            else:
                                # Candidate returned to forward gaze
                                gaze_away_since = None
                                gaze_away_logged = False

            # =========================================
            # Face not detected
            # =========================================

            else:
                gaze_history.clear()
                closed_since = None
                gaze_away_since = None

                gaze_away_logged = False
                eyes_closed_logged = False

                label = "NO FACE DETECTED"
                color = (80, 80, 255)

            # =========================================
            # Calibration using C
            # =========================================

            if key in (ord("c"), ord("C")):

                if (
                    result.face_landmarks
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

                    label = "CALIBRATED - LOOK FORWARD"
                    color = (100, 220, 140)

                    print(
                        "Calibration complete:",
                        f"x={baseline_x:.3f}, "
                        f"y={baseline_y:.3f}",
                    )

                else:
                    print(
                        "Calibration skipped: keep your face "
                        "visible and eyes open."
                    )

            # =========================================
            # Display webcam status
            # =========================================

            cv2.rectangle(
                frame,
                (12, 12),
                (480, 65),
                (25, 25, 25),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (22, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "AI Proctoring - Calibrated Gaze",
                frame,
            )

            if key in (ord("q"), ord("Q")):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Eye tracking closed.")