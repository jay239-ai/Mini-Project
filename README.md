# Free-Left Traffic Dashboard

A prototype for analyzing three recorded traffic clips or live USB and supported IP camera feeds, importing historical traffic PDFs, and producing an explainable junction advisory. The advisory is for human review; it does **not** make certified safety decisions or control real traffic lights.

## Run it in VS Code

1. Open this folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a Python environment (first time only):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```powershell
   $env:PYTHONPATH = "$PWD\backend;$PWD"
   python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8765 --reload
   ```

5. Open <http://127.0.0.1:8765/dashboard.html>. Choose recorded clips, wired USB cameras, or supported network camera streams, then use the same site name when importing historical reports. Keep the terminal running while using the page. Press **Ctrl+C** to stop the app.

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## What the three views do

- **Camera 01 — main approach/free-left:** vehicle detections and approach/free-left measurements.
- **Camera 02 — free-left turn:** turning-lane and pedestrian conflict observations.
- **Camera 03 — right-side joining road:** moving/stationary vehicle flow, stopped queue and blockage observations on the road feeding into the free-left route.

The three uploaded clips are analyzed concurrently. Camera-local detections are estimates, not ground truth; the cross-camera pass attempts conservative de-duplication. The advisory uses configured thresholds and names the evidence and rule that triggered it.

The dashboard provides separate **Upload videos** and **Live cameras** interfaces with the three-camera 16:9 analysis grid at the top. Both show clear frames with tracking/ROI overlays; camera images are processed locally and are not recorded to disk. Capture and preview refresh request up to 30 FPS. The actual detector rate depends on the camera, hardware, resolution and scene density, and appears per camera; a 30 FPS request is not a hardware guarantee. **Fast mode** is the default: CPU uses 256-pixel inference and analyzes every decoded upload frame; GPU uses 416-pixel inference (512 for turning) and every frame. Live cameras request 30 FPS and process the newest available frame without building a delay queue. **Balanced mode** uses larger model sizes and CPU stride of two for uploads. Fast mode can miss small or briefly visible vehicles and may fragment tracks more often; use Balanced mode when detail is more important than response time. If the real-time factor is below 1.0×, uploaded-clip analysis finishes later than clip playback. The traffic advisory refreshes from current camera measurements several times per second; annotated previews target 30 FPS. **Stop analysis** ends either mode after the current inference call, finalizes a partial-session PDF, and requests its download. The report button remains available as a fallback. If stopped before any frames are processed, the PDF states that no camera measurements were available.

The dashboard's **Live cameras** mode scans locally attached USB webcam devices (DirectShow, Media Foundation, then the default OpenCV backend); on Android, enable the phone's USB webcam mode before scanning. It also accepts RTSP/RTSPS/HTTP/HTTPS stream URLs for network cameras and other OpenCV-readable live sources. The live capture reader continually overwrites a one-frame buffer so it analyzes the newest available image rather than accumulating a stale queue. Live previews show clear frames with tracking boxes and measured FPS/frame-age telemetry. The green/yellow/red indicator is a traffic-condition advisory, not an instruction to change a real signal. Green requires low measured pressure and the minimum observation window; yellow covers medium pressure or incomplete evidence; red flags high pressure or an observed pedestrian/vehicle conflict. One source can be previewed and tracked. Three distinct views are required for a full junction advisory.

Cross-camera de-duplication is an experimental, conservative second pass. It compares transient per-track HSV color histograms and clip timestamps, then only merges high-similarity tracks with overlapping time intervals and no similarly scoring competing track. The camera clock offsets in `config/camera_zones.example.json` can align known timestamp differences. Appearance features stay in memory for the job and are discarded; source uploads are removed when analysis ends, and live frames are not recorded. This avoids plate recognition and persisted vehicle images. Similar colored vehicles, major viewpoint changes, occlusion, low light, and unsynchronized footage can still cause missed or false matches. The output is therefore an estimate after accepted matches plus the per-camera total as an upper bound, not a guaranteed exact identity count.

## Decision logic and PDF history

The advisory first checks that all three named camera roles are present, each has valid measurements, each role has its required ROI kinds, and live measurements are no older than 2.5 seconds. Queue and flow decisions begin after two seconds of observation; green additionally needs at least 30 processed frames from each view. A person in a configured free-left or pedestrian-conflict area turns the advisory red immediately and keeps it red for a two-second clearance hold, even before the other cameras finish connecting. The signal is recalculated as measurements arrive (up to five times per second); the live display polls more often for fresh frames and counts. A sustained stopped joining-road queue raises yellow; a large or long-lasting stopped queue raises red. Uploaded-video decisions update during analysis as provisional measurements arrive. Missing, duplicated, malformed, stale, too-short, or incorrectly configured inputs return **insufficient data** instead of a guessed recommendation.

Current evidence is combined from the 95th-percentile queue count (to reduce one-frame spikes), average stopped delay per tracked vehicle, the longest blockage event (rather than adding overlapping blockage times), and pedestrian/vehicle co-presence in configured conflict ROIs. The explainable score weights queue pressure up to 40 points, stopped delay up to 25, blockage up to 20, and conflict evidence up to 25. Thresholds in `config/camera_zones.example.json` set medium/high queue, delay, blockage, and score boundaries. Results include score components, confidence, data-quality warnings, and the triggered rule. Confidence is capped at moderate until validated against reviewed site data. No phase duration is generated and no signal command is sent.

The report panel accepts text-based PDFs with labeled traffic fields such as jam/incident count, violation count, congestion level, queue size/length, delay duration, and reporting dates. Text extraction runs locally. Scanned/image-only PDFs, password-protected PDFs, malformed files, and unrecognized reports are rejected with a reason; OCR is not enabled. Report/site facts are kept in a local SQLite database at `data/processed/traffic_history.sqlite3` (created automatically using Python's built-in SQLite; no database server is needed). The original PDF, extracted text, and user filename are discarded. Reports are matched by the site name entered in the dashboard; exact file duplicates are ignored. Counts are shown per report/latest report to avoid summing overlapping reporting periods. Only readable reports with a recognized date within the last 365 days can add a capped historical watch factor. History may add at most 10 score points and can shift only near-threshold current evidence from green to yellow; it cannot independently produce red. Old or undated reports remain visible as historical context without influencing the current advisory.

The `PyMuPDF` package added to `requirements.txt` extracts PDF text. This adds one package to the existing local Python install; the database itself is part of Python. The bundled `yolo11n.pt` is the detector model file. No cloud service receives camera frames or PDF contents. The sample polygons in `config/camera_zones.example.json` still need to be drawn over the real main approach, free-left turn, pedestrian conflict area and right-side joining road; without that site calibration, movement estimates and signal colors are not reliable enough for real traffic operation.

The generated session PDF includes per-camera vehicle and pedestrian track counts, vehicle observations by configured ROI, the final advisory, and a signal timeline with measured update intervals for sessions analyzed by the current version. Older saved sessions did not store intermediate signal states; their reports show that transition history is unavailable rather than estimating it after the fact. The optional `pedestrian_nearby` region in camera 2 is a buffer for vehicles near the crossing; only a pedestrian inside the crossing ROI together with a vehicle in the crossing or nearby buffer creates a pedestrian/vehicle overlap sample. The sample polygon must be calibrated to the real camera.

Queue span from camera footage is still an image-pixel estimate unless a calibrated road-plane transform is configured. A queue length stated inside a historical report is displayed as a source-reported value and is not treated as a camera-calibrated measurement.

## Project map

- `frontend/` — dashboard page, styling, and browser behavior.
- `backend/app/` — web server, upload/progress endpoints, and PDF summary.
- `vision_engine/` — object detection, tracking, regions of interest, and traffic measurements.
- `config/camera_zones.example.json` — sample regions and thresholds; these need calibration for your cameras.
- `yolo11n.pt` — object-detection model used by the prototype.
- `vision_engine/metrics/cross_camera.py` — privacy-preserving appearance/time association for cross-camera vehicle tracklets.
- `backend/app/decision_engine/rules.py` — quality-gated, explainable traffic advisory rules.
- `backend/app/reporting/historical_reports.py` — local PDF extraction and SQLite history storage.
- `data/processed/traffic_history.sqlite3` — automatically created local historical summary database.

## Current limits

Live USB operation requires Windows to expose each attached phone/camera as a webcam device; network streams need an address and format supported by the installed OpenCV backend. Camera credentials in a stream URL are used only by the local capture process and are not included in session results. The detector dependencies must be installed. The ROI polygons are examples and must be calibrated to each camera view before treating zone measurements as meaningful. Uploaded camera clips are temporary; analysis summaries and generated reports are written under `data/processed/`. Tracking preview frames are transient and are not saved as video files. The rules are explainable and defensive, not perfect; their thresholds require validation against labeled footage and site-specific review before they can inform real operations.
