
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise ValueError(
        "DATABASE_URL not found. Check your .env file."
    )

try:
    with psycopg.connect(
        database_url,
        connect_timeout=10,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT current_database(), NOW();"
            )
            database_name, server_time = cur.fetchone()

    print("Supabase connection successful!")
    print("Database:", database_name)
    print("Server time:", server_time)

except Exception as error:
    print("Connection failed.")
    print("Error type:", type(error).__name__)
    print("Please check the connection settings.")
