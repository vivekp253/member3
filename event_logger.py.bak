
import os
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def save_proctoring_event(
    exam_id,
    event_type,
    severity="warning",
    duration_ms=None,
    details=None,
):
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing from .env"
        )

    query = """
        INSERT INTO public.proctoring_events
            (exam_id, event_type, severity, duration_ms, details)
        VALUES
            (%s, %s, %s, %s, %s)
        RETURNING id, occurred_at;
    """

    with psycopg.connect(
        DATABASE_URL,
        connect_timeout=10,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    exam_id,
                    event_type,
                    severity,
                    duration_ms,
                    Jsonb(details or {}),
                ),
            )
            event_id, occurred_at = cur.fetchone()

    return {
        "id": event_id,
        "occurred_at": occurred_at.isoformat(),
        "event_type": event_type,
    }
