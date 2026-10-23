# AI Exam Proctoring System

An AI-powered online examination monitoring system built using computer vision, browser activity monitoring, Flask, and Supabase. The system is designed to detect potentially suspicious examination activities, record monitoring events, and support administrators in reviewing examination sessions.

## About the Project

Online examinations can make it difficult to continuously monitor every candidate. This project aims to assist examination supervision through automated monitoring of webcam input and browser activity.

The system contains modules for face detection, eye and gaze tracking, mobile phone detection, browser tab-switch monitoring, and event logging. Recorded events can be stored in Supabase for later retrieval and review.

The project is designed to support integration with larger examination platforms, including AI-powered interview and candidate evaluation systems.

## Key Features

### 1. Face Detection
- Detects faces in the webcam feed.
- Supports monitoring for face absence and multiple faces, according to the configured implementation.
- Can generate events for later review.

### 2. Eye and Gaze Tracking
- Uses facial landmarks to estimate gaze direction.
- Helps identify gaze deviations that may warrant review.
- Performance depends on lighting, camera placement, and facial visibility.

### 3. Mobile Phone Detection
- Uses a YOLO object detection model to detect mobile phones visible to the camera.
- Detection accuracy depends on the camera angle, lighting, object visibility, and model configuration.

### 4. Browser Tab-Switch Monitoring
- Uses JavaScript browser visibility monitoring to detect when the examination page becomes hidden.
- Sends monitoring events to the backend when configured.
- Can support examination rules based on tab-switch counts if implemented in the application.

### 5. Event Logging
- Records monitoring events through the event logger.
- Supports storing event details in Supabase.
- Allows events to be retrieved for examination review.

### 6. Flask Backend
- Provides backend functionality for examination monitoring.
- Handles supported API requests and event logging.
- Acts as the integration point for other applications.

### 7. Admin Dashboard
- Includes an HTML dashboard template for presenting examination monitoring information.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Flask | Backend API |
| OpenCV | Image and webcam processing |
| MediaPipe | Face detection and facial landmark tracking |
| YOLO | Object detection and phone monitoring |
| JavaScript | Browser tab-switch monitoring |
| HTML/CSS | Dashboard and testing interfaces |
| Supabase | Cloud database and event storage |

## Project Structure

The following structure is based on the files currently present in the repository.

```text
AI-Exam-Proctoring/
│
├── models/
│   ├── blaze_face_short_range.tflite
│   └── face_landmarker.task
│
├── templates/
│   └── admin_dashboard.html
│
├── app.py
├── event_logger.py
├── event_logger.py.bak
├── eye_tracking.py
├── face_detection.py
├── face_detection.py.bak
├── integrated_monitor.py
├── phone_detection.py
├── phone_detection.py.bak
├── proctoring_backup_faf0ab09-514e-4f8c-8c95-f6200ba30e69.json
├── proctoring_code.txt
├── project-files.txt
├── tab_monitor.js
├── test_db.py
├── test_event.py
├── test_tab.html
├── webcam_test.py
├── yolo11n.pt
└── .gitignore
```

The `.env` file is intentionally not listed in this documentation.

## Understanding the Important Files

| File or folder | Description |
|---|---|
| `app.py` | Flask backend and API routes |
| `event_logger.py` | Handles proctoring event logging |
| `eye_tracking.py` | Eye and gaze tracking logic |
| `face_detection.py` | Face detection logic |
| `integrated_monitor.py` | Intended to support integrated monitoring |
| `phone_detection.py` | Mobile phone detection |
| `tab_monitor.js` | Browser tab visibility monitoring |
| `models/` | Contains face detection and facial landmark models |
| `templates/admin_dashboard.html` | Admin dashboard interface |
| `yolo11n.pt` | YOLO model weights |
| `test_db.py` | Database connectivity testing |
| `test_event.py` | Event logging testing |
| `test_tab.html` | Browser tab monitoring testing |
| `webcam_test.py` | Webcam testing |
| `.gitignore` | Defines files Git should ignore |

The `.bak` files are backups, while the JSON, text, and file-listing artifacts are additional project files. Their exact contents should be reviewed before deciding whether they are needed by the application.

## Prerequisites

Before running the project, install or obtain access to:

- Python compatible with the project's dependencies.
- Git, if cloning the repository.
- Visual Studio Code or another code editor.
- A working webcam for camera-based monitoring.
- Internet access for Supabase connectivity.
- Access to the configured Supabase project.
- The required Python packages and machine learning model files.

Verify your installations:

```bash
python --version
git --version
```

## Getting the Project

### Option 1: Clone the Repository

Open a terminal and run:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Exam-Proctoring
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the repository's actual URL.

For a private repository, sign in to GitHub using an authorized account and accept the collaborator invitation if required.

### Option 2: Download the ZIP

1. Open the repository on GitHub.
2. Click the green **Code** button.
3. Select **Download ZIP**.
4. Extract the downloaded ZIP file.
5. Open the extracted project folder in Visual Studio Code.

Ensure that you open the directory containing `app.py`.

## Installation and Setup

### Step 1: Create a Virtual Environment

Open the project folder in VS Code and select **Terminal → New Terminal**.

On Windows PowerShell, run:

```powershell
python -m venv .venv
```

A virtual environment keeps this project's Python dependencies separate from those of other projects.

### Step 2: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

After activation, the terminal generally displays `(.venv)` at the beginning of the prompt.

### Step 3: Install Dependencies

If the repository contains a verified `requirements.txt`, run:

```bash
python -m pip install -r requirements.txt
```

If that file is not available, obtain the correct dependency list from the project maintainer before proceeding.

Common packages used by projects with this technology stack include Flask, OpenCV, MediaPipe, Ultralytics, and the Supabase Python client. Install only the packages required by the actual source code, using compatible versions.

### Step 4: Check Model Files

Verify that these files are present:

- `models/blaze_face_short_range.tflite`
- `models/face_landmarker.task`
- `yolo11n.pt`

The model paths used in the source code must match the files' actual locations.

### Step 5: Check Application Configuration

The application requires access to its configured database and other required settings.

Use the existing authorized project configuration provided by the project maintainer. Confirm that the Supabase project is accessible and that the expected database tables and columns exist.

## Running the Application

### Step 1: Start the Flask Backend

Activate the virtual environment and run:

```bash
python app.py
```

Wait for the server startup message. Keep the terminal open while testing the application.

If startup fails, inspect the error message and resolve missing packages, configuration issues, or port conflicts.

### Step 2: Open the Backend

Use the host and port specified in the Flask application.

If the backend is configured to run locally on port `5000`, the address may be:

```text
http://127.0.0.1:5000
```

The root URL may not display a webpage unless a root route is defined.

### Step 3: Open the Admin Dashboard

The project contains the dashboard template:

```text
templates/admin_dashboard.html
```

Inspect `app.py` to identify the route that renders the dashboard, then open that route in your browser.

### Step 4: Test Individual Components

Run and test the individual modules according to their implementation. Verify each component separately before testing the integrated monitoring workflow.

## How the System Works

The intended examination monitoring workflow is:

1. A candidate begins an examination session.
2. The application starts the configured monitoring components.
3. The face detection component analyzes webcam input.
4. The gaze tracking component estimates eye direction from facial landmarks.
5. The phone detection component searches for mobile phones in the camera view.
6. Browser monitoring observes examination-page visibility changes.
7. Detected events are passed to the event logger.
8. The backend stores supported event records in Supabase.
9. An administrator retrieves and reviews the recorded examination activity.

The precise startup sequence and whether all components run simultaneously depend on the current implementation.

## Supabase Integration

Supabase is used for storing proctoring events.

The database configuration and table schema must match what the application expects.

To verify database integration:

1. Confirm that the configured Supabase project is accessible.
2. Check that the required tables and columns exist.
3. Run the database test script if it is configured for that purpose.
4. Run the event logging test script.
5. Inspect the database to confirm that test events were saved.

### Testing Scripts

| Script | Intended purpose |
|---|---|
| `test_db.py` | Tests database connectivity |
| `test_event.py` | Tests event logging |

Check each script's implementation to determine the exact command and expected output.

## Integrating with TalentIQ or Another Application

This proctoring system can be integrated with an interview or examination platform.

For example, TalentIQ can manage candidates, interview questions, submitted answers, and evaluation scorecards, while this project manages proctoring events.

### Integration Workflow

**Step 1 — Create an exam session**

The main application creates or identifies the examination session and assigns it a unique `exam_id`.

**Step 2 — Identify the candidate**

Associate the examination with the appropriate `candidate_id`.

**Step 3 — Start proctoring**

The main application initializes the proctoring functionality using the supported backend routes and request formats.

**Step 4 — Record monitoring events**

During the examination, the proctoring components generate supported events, such as tab switches, face absence, multiple faces, or phone detections.

**Step 5 — Submit interview answers**

The interview application submits the candidate's actual answers to the evaluation service.

**Step 6 — Evaluate the interview**

The evaluation agent processes the submitted answers and generates a scorecard containing scores and feedback.

**Step 7 — Display the results**

The admin dashboard retrieves the evaluation scorecard and proctoring events associated with the same examination session.

### Important Integration Requirements

- Use a consistent `exam_id` across the interview evaluation and proctoring records.
- Use `candidate_id` to identify the candidate.
- Assign a new `exam_id` to a new examination attempt.
- Ensure that API routes and request payloads match the actual backend implementations.
- Verify database permissions and access controls.
- Test the complete workflow using synthetic candidate data before production use.

The projects do not necessarily need to share a single physical database. Separate tables or databases can be used as long as the application reliably correlates the records.

This section describes the intended integration design; developers must verify which connections are already implemented in the current codebase.

## Testing the Complete Workflow

Before using the system with real candidates, verify the following:

- The Flask backend starts without errors.
- Webcam access works correctly.
- Face detection behaves as expected.
- Gaze tracking produces usable results.
- Phone detection works under the intended conditions.
- Browser tab-switch events reach the backend.
- Events are saved to the intended Supabase project.
- The admin dashboard retrieves the correct examination events.
- The interview evaluation is based on actual submitted answers.
- The evaluation scorecard and proctoring events correspond to the same exam.

Use test candidates and synthetic examination data during development.

## Troubleshooting

### Python or Package Errors

Confirm that Python is installed, the virtual environment is activated, and all required dependencies are installed.

### Model File Not Found

Verify that the model file exists at the path specified in the source code.

### Webcam Not Working

Check operating-system camera permissions, close other applications using the webcam, and verify that the correct camera device is selected.

### Face Detection Is Inconsistent

Check lighting, camera placement, model paths, and detection thresholds.

### Phone Detection Is Inconsistent

Check the model weights, confidence threshold, camera angle, and visibility of the phone.

### Events Are Not Saved

Verify database connectivity, table names, column names, permissions, and event logger behavior.

### Tab-Switch Events Are Missing

Check that the JavaScript is loaded by the examination page and that the configured backend route is available.

### Dashboard Does Not Display Events

Verify the dashboard route, API response, database records, and examination identifiers.

## Security and Privacy

- Restrict repository access to authorized collaborators.
- Do not expose API keys, database credentials, or other secrets in source code.
- Do not commit private candidate information or real examination records.
- Obtain appropriate consent before webcam monitoring.
- Apply suitable access controls to the database.
- Review AI-generated monitoring alerts in context rather than treating them as definitive proof of misconduct.

## Future Enhancements

- Integrate all monitoring components into a unified workflow.
- Improve phone detection from different camera angles.
- Improve gaze tracking under varied lighting conditions.
- Enhance real-time event reporting.
- Integrate interview evaluation and proctoring results into a unified dashboard.
- Add automated tests and deployment documentation.

## Contributing

Contributions and suggestions are welcome.

1. Discuss proposed changes with the project team.
2. Create a separate branch for your changes.
3. Test the affected components.
4. Submit a pull request describing the changes.

## License

A license has not yet been specified. Add an appropriate license before distributing the project for reuse.

---

**AI Exam Proctoring System**  
AI-assisted online examination monitoring using computer vision, browser activity tracking, and event logging.
