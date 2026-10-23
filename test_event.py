
from event_logger import save_proctoring_event

# Replace this with the UUID returned by Supabase
EXAM_ID = "56d45458-b61b-4ee4-b43f-832eedfff6f6"

try:
    result = save_proctoring_event(
        exam_id=EXAM_ID,
        event_type="face_absent",
        severity="warning",
        duration_ms=15000,
        details={
            "message": "No face detected for 15 seconds",
            "source": "test_event.py",
        },
    )

    print("Event saved successfully!")
    print("Event ID:", result["id"])
    print("Event type:", result["event_type"])
    print("Occurred at:", result["occurred_at"])

except Exception as error:
    print("Could not save event.")
    print("Error type:", type(error).__name__)
    print("Error:", str(error))
