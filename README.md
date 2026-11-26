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

5. Open <http://127.0.0.1:8765/dashboard.html>. Choose recorded clips, wired USB cameras, or supported network streams. Keep the terminal running while using the page. Press **Ctrl+C** to stop the app.

## Camera roles and decision behavior

- **Camera 1 — main approach and free-left:** calibrate a main-approach polygon that excludes the opposite lane, then a free-left polygon inside it. The dashboard reports confirmed tracks, current free-left queue, queue span, and stopped delay. A busy approach produces red only when camera 3 also sees heavy moving flow.
- **Camera 2 — conflict path:** drawing a `turn_path` polygon is optional. Use the dashboard preview tool or `config/camera_zones.example.json` to mark the vehicle-accessible area. With a polygon, only pedestrians and stopped vehicles inside it affect the decision; without one, the whole frame is analyzed. Pedestrians in the checked area trigger red immediately; vehicles trigger a hold only after remaining stationary there for the configured blockage interval. Moving vehicles, including ones approaching the camera, do not trigger red just for being visible. Person detections overlapping a bicycle or motorcycle detection are treated as riders rather than pedestrians.
- **Camera 3 — joining road:** the full frame measures moving and stationary vehicle flow. High moving flow can raise red when the free-left path is also busy. Stationary vehicles alone raise caution because they may be waiting at their own signal.

Camera 1’s sample polygons are illustrative and must be adjusted to the actual camera view. Camera 2’s polygon is optional: without one, pedestrians anywhere in its full frame are treated as a path obstruction. Camera 3 continues to use its full frame.

## Preview and reports

The **Upload videos** and **Live cameras** screens show tracking boxes, camera status, and measured processing speed. Live capture uses a one-frame buffer so it analyzes the newest available image. Capture requests up to 30 FPS, but actual detector speed depends on hardware, resolution, and scene density. Balanced mode is the default; Fast mode trades detail for response time and may miss small or brief vehicles or fragment tracks.

**Stop analysis** waits for the current inference step, finalizes a PDF from the observations gathered so far, and requests the download. The report button remains available as a fallback. Very short sessions may end with insufficient evidence. Camera images are processed locally; preview frames are transient and not saved as video.

## Counting and measurement limits

A track must appear in two analyzed frames before it contributes to confirmed totals. Confidence-weighted class votes stabilize labels, and a conservative short-gap box match can bridge some brief tracker ID breaks. These reduce one-frame false detections, class flicker, and duplicate fragments, but occlusion, dense traffic, camera shake, low light, riders, and overlapping vehicles can still cause missed or duplicate detections. Counts stay separate by camera role; the same vehicle may appear in more than one view, so do not add them into a junction-wide total. Cross-camera appearance matching remains experimental and is not treated as ground truth.

Queue span is measured in image pixels and normalized against the free-left ROI’s image diagonal for the decision threshold. It is not physical distance; perspective and polygon calibration affect it. Accurate meters require a road-plane calibration. The bundled `yolo11n.pt` model is small and generic. Validate thresholds and counts against manually reviewed site footage before using them for any real-world traffic operation.

## Historical reports and privacy

Text-based PDFs with labeled traffic fields can be imported. Scanned/image-only, password-protected, malformed, or unrecognized PDFs are rejected; OCR is not enabled. Text extraction and camera processing run locally. Site summaries are stored in `data/processed/traffic_history.sqlite3`; original PDFs, extracted text, and filenames are discarded. Historical reports are shown as context only and do not change the current camera-based signal indication.

Camera credentials in stream URLs are used only by local capture and are not included in session results. Network cameras must use a format supported by the installed OpenCV backend. Live USB operation requires Windows to expose each attached camera or phone as a webcam device. Opening a phone's regular camera app does not make its video available to the computer; enable the phone's USB webcam/UVC mode or use a phone streaming app and enter its RTSP/HTTP URL. USB discovery assigns detected devices to the first camera slots automatically.

## Project map

- `frontend/` — dashboard page and browser behavior.
- `backend/app/` — web server, analysis endpoints, decision rules, and PDF summaries.
- `vision_engine/` — detection, tracking, regions, and traffic measurements.
- `config/camera_zones.example.json` — sample camera polygons and thresholds.
- `backend/tests/test_camera_decision_logic.py` — role-specific decision and tracking regression checks.
