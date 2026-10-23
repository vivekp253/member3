
const MAX_ALLOWED_SWITCHES = 2;
const API_URL = "http://127.0.0.1:5000/tab-switch";

// Replace with an existing exam_id UUID from Supabase
const EXAM_ID = "8b2b1b5e-220a-4f53-b575-75fcd4d81445";

let hiddenSince = null;
let switchCount = 0;
let examClosed = false;

const status = document.getElementById("status");
const log = document.getElementById("log");

document.addEventListener("visibilitychange", async () => {
    if (examClosed) return;

    if (document.hidden) {
        if (hiddenSince === null) {
            hiddenSince = Date.now();
            switchCount++;

            console.warn(`Tab switch ${switchCount}`);

        }
        return;
    }

    if (hiddenSince === null) return;

    const durationMs = Date.now() - hiddenSince;
    hiddenSince = null;

    // Send the event to the Python backend
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                exam_id: EXAM_ID,
                switch_count: switchCount,
                duration_ms: durationMs
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Request failed");
        }

        console.log("Tab-switch event saved:", result);

    } catch (error) {
        console.error(
            "Could not send tab-switch event:",
            error
        );
    }

    if (switchCount >= MAX_ALLOWED_SWITCHES) {
        examClosed = true;

        if (status) {
            status.textContent =
                "EXAM CLOSED — Tab switched 2 times.";
            status.style.color = "red";
        }

        if (log) {
            log.textContent =
                `Tab switches: ${switchCount}/2\n` +
                `Last hidden duration: ` +
                `${(durationMs / 1000).toFixed(1)} seconds\n` +
                "Exam terminated due to the tab-switch limit.";
        }

        document.body.style.pointerEvents = "none";
        document.body.style.userSelect = "none";

        console.error("EXAM CLOSED: Tab-switch limit reached.");

    } else {
        if (status) {
            status.textContent =
                `Tab switches: ${switchCount}/2`;
            status.style.color = "#16803d";
        }

        if (log) {
            log.textContent =
                `Switch count: ${switchCount}/2\n` +
                `Last hidden duration: ` +
                `${(durationMs / 1000).toFixed(1)} seconds`;
        }
    }
});
