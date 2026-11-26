"""PDF report builder for persisted/serialized session analysis results."""

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def generate_session_pdf(session: dict[str, Any], output_path: Path) -> Path:
    """Create a readable PDF using only values present in a session summary."""

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as error:
        raise RuntimeError("Install project dependencies to generate PDF reports") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#16324f"), spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#4b5563")))
    story: list[Any] = [
        Paragraph("Free-Left Traffic Analysis Report", styles["ReportTitle"]),
        Paragraph(f"Site: {escape(_pdf_safe(session.get('site_label', 'not recorded')))} | Mode: {escape(_pdf_safe(session.get('session_mode', 'unknown')))} | Generated: {escape(_pdf_safe(session.get('generated_at', 'not recorded')))}", styles["BodyText"]),
        Paragraph(f"Data coverage: {escape(str(session.get('data_status', 'unknown')))} | Cameras analyzed: {escape(str(session.get('camera_count', 'unknown')))}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Counts are confirmed camera-local tracks after a two-frame minimum. Camera 1 counts vehicles entering its free-left polygon; camera 3 counts the full joining-road view. The same physical vehicle may appear in both views, so these role counts are not added into a junction-wide vehicle total.", styles["SmallNote"]),
        Spacer(1, 10),
        Paragraph("Camera results", styles["Heading2"]),
    ]
    rows = [["Camera", "Frames", "Coverage", "Relevant vehicle tracks", "Scope", "Max occupancy", "Avg wait", "Blocked s", "Blockage %"]]
    category_rows = [["Camera role", "Category", "Confirmed relevant tracks"]]
    for camera in session.get("camera_results", []):
        camera_categories = camera.get("relevant_vehicle_category_tracks", camera.get("vehicle_category_tracks", {}))
        vehicle_tracks = int(camera.get("relevant_vehicle_track_count", camera.get("vehicle_track_count", sum(int(value) for value in camera_categories.values()))))
        zone_tracks = camera.get("zone_vehicle_tracks", {})
        camera_id = str(camera.get("camera_id", "unknown"))
        if camera_id == "approach":
            scope = f"Main lane {camera.get('approach_lane_vehicle_track_count', 0)} · free-left {zone_tracks.get('free_left_lane', vehicle_tracks)} tracks"
        elif camera_id == "turning":
            scope = f"Full-frame turn path · blocked frames {camera.get('turn_path_obstruction_frames', 0)}; max blockage {camera.get('turn_path_max_obstruction_count', 0)}"
        else:
            scope = f"Full-frame joining road · max moving {camera.get('joining_flow_max_moving_vehicles', 0)} / stopped {camera.get('joining_flow_max_stationary_vehicles', 0)}"
        rows.append([
            camera_id,
            _display(camera.get("frame_count_processed")),
            _display(camera.get("frame_coverage_percent"), "%"),
            str(vehicle_tracks),
            Paragraph(escape(_pdf_safe(scope)), styles["SmallNote"]),
            _display(camera.get("maximum_queue_vehicles")),
            _display(camera.get("average_wait_seconds"), " s"),
            _display(camera.get("total_blocked_seconds"), " s"),
            _display(camera.get("blockage_ratio_percent"), "%"),
        ])
    camera_table = Table(rows, repeatRows=1, colWidths=[15 * mm, 12 * mm, 13 * mm, 23 * mm, 42 * mm, 13 * mm, 16 * mm, 15 * mm, 15 * mm])
    camera_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c2cc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([camera_table, Spacer(1, 8)])

    camera_errors = session.get("camera_errors")
    camera_warnings = session.get("camera_warnings")
    if (isinstance(camera_errors, dict) and camera_errors) or (isinstance(camera_warnings, dict) and camera_warnings):
        story.append(Paragraph("Camera input notes", styles["Heading3"]))
    if isinstance(camera_errors, dict):
        for camera_id, message in sorted(camera_errors.items()):
            story.append(Paragraph(f"{escape(_pdf_safe(camera_id))} error: {escape(_pdf_safe(message))}", styles["SmallNote"]))
    if isinstance(camera_warnings, dict):
        for camera_id, messages in sorted(camera_warnings.items()):
            for message in messages:
                story.append(Paragraph(f"{escape(_pdf_safe(camera_id))}: {escape(_pdf_safe(message))}", styles["SmallNote"]))
    if (isinstance(camera_errors, dict) and camera_errors) or (isinstance(camera_warnings, dict) and camera_warnings):
        story.append(Spacer(1, 6))

    story.append(Paragraph("Vehicle category counts by relevant camera view", styles["Heading2"]))
    for camera in session.get("camera_results", []):
        cats = camera.get("relevant_vehicle_category_tracks", camera.get("vehicle_category_tracks", {}))
        role = str(camera.get("display_name", camera.get("camera_id", "camera")))
        if cats:
            category_rows.extend([[role, name, str(count)] for name, count in sorted(cats.items())])
    if len(category_rows) == 1:
        category_rows.append(["No confirmed relevant tracks", "—", "0"])
    category_table = Table(category_rows, colWidths=[72 * mm, 43 * mm, 45 * mm])
    category_table.setStyle(_simple_table_style(colors))
    story.extend([category_table, Spacer(1, 8)])

    story.append(Paragraph("Vehicle observations by camera region", styles["Heading2"]))
    zone_rows = [["Camera", "Relevant confirmed tracks", "Counting / safety scope"]]
    for camera in session.get("camera_results", []):
        zone_tracks = camera.get("zone_vehicle_tracks", {})
        camera_id = str(camera.get("camera_id", "unknown"))
        if camera_id == "approach":
            zone_text = (
                f"Main-lane polygon: {camera.get('approach_lane_vehicle_track_count', 0)} confirmed tracks; "
                f"free-left subregion: {zone_tracks.get('free_left_lane', 0)} confirmed tracks. "
                f"Current queue span {camera.get('current_queue_extent_pixels', 0)} px; maximum observed span "
                f"{camera.get('maximum_queue_extent_pixels', 0)} px; stopped delay {camera.get('total_stopped_delay_seconds', 0)} s. "
                f"P95 span ratio against the free-left ROI diagonal: {camera.get('p95_queue_length_ratio', 0)}. "
                "Pixel spans are image-space estimates, not physical distances."
            )
        elif camera_id == "turning":
            zone_text = (
                f"Scope: {'configured turn-path polygon' if camera.get('turn_path_zone_configured') else 'full frame (no polygon configured)'}. "
                f"Pedestrians in the checked area block immediately. Vehicles count as obstruction after remaining stopped "
                f"in the checked area for the configured interval. Blocked in {camera.get('turn_path_obstruction_frames', 0)} frames."
            )
        else:
            zone_text = "No polygon. The full frame is used for joining-road vehicle counts and movement classification."
        zone_rows.append([
            _pdf_safe(camera.get("display_name", camera.get("camera_id", "unknown"))),
            str(camera.get("relevant_vehicle_track_count", camera.get("vehicle_track_count", 0))),
            Paragraph(escape(zone_text), styles["SmallNote"]),
        ])
    zone_table = Table(zone_rows, repeatRows=1, colWidths=[48 * mm, 28 * mm, 94 * mm])
    zone_table.setStyle(_simple_table_style(colors))
    story.extend([
        zone_table,
        Spacer(1, 4),
        Paragraph(
            "Camera counts stay separate by role. Do not add camera 1 and camera 3 counts to estimate a unique junction vehicle total; camera-to-camera identity matching is not reliable enough to claim that value.",
            styles["SmallNote"],
        ),
        Spacer(1, 8),
    ])

    story.append(Paragraph("Frame-level traffic and image-space occupancy", styles["Heading2"]))
    occupancy_rows = [["Camera", "Avg vehicles/frame", "Peak vehicles/frame", "Avg box area %", "Peak box area %", "Inference FPS"]]
    for camera in session.get("camera_results", []):
        occupancy_rows.append([
            str(camera.get("camera_id", "unknown")),
            _display(camera.get("average_visible_vehicles_per_frame")),
            _display(camera.get("maximum_visible_vehicles_in_frame")),
            _display(camera.get("average_vehicle_bounding_box_area_percent"), "%"),
            _display(camera.get("maximum_vehicle_bounding_box_area_percent"), "%"),
            _display(camera.get("inference_fps")),
        ])
    occupancy_table = Table(occupancy_rows, repeatRows=1, colWidths=[22 * mm, 31 * mm, 31 * mm, 27 * mm, 27 * mm, 25 * mm])
    occupancy_table.setStyle(_simple_table_style(colors))
    story.extend([
        occupancy_table,
        Spacer(1, 4),
        Paragraph("Bounding-box area is an approximate image-space proxy. Boxes may overlap; this is not calibrated physical road occupancy.", styles["SmallNote"]),
        Spacer(1, 8),
    ])

    history = session.get("historical_context", {})
    story.append(Paragraph("Historical context", styles["Heading2"]))
    if history.get("matched_site"):
        story.append(Paragraph(
            f"Reports for {escape(_pdf_safe(history.get('site_label', session.get('site_label', 'selected site'))))}: {escape(str(history.get('report_count', 0)))} report(s); reported jams: {escape(str(history.get('incident_count', 'not available')))}; violations: {escape(str(history.get('violation_count', 'not available')))}; peak congestion: {escape(str(history.get('peak_congestion_level', 'not recorded')))}; largest reported queue: {escape(str(history.get('maximum_queue_vehicles', 'not recorded')))} vehicles / {escape(str(history.get('maximum_queue_length_meters', 'not recorded')))} meters; average delay: {escape(str(history.get('average_delay_minutes', 'not recorded')))} minutes.",
            styles["BodyText"],
        ))
        story.append(Paragraph("Historical report data is used as a capped context factor only for dated, readable reports within the last 365 days. It may shift near-threshold current pressure from green to yellow, but cannot produce red by itself. Counts use the latest report to avoid summing overlapping periods. The original PDF and extracted text are not retained.", styles["SmallNote"]))
    else:
        story.append(Paragraph("No historical report matched this site label. The advisory uses current camera measurements only.", styles["BodyText"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Junction advisory", styles["Heading2"]))
    decision = session.get("junction_decision", {})
    traffic_light = decision.get("traffic_light", {})
    story.extend([
        Paragraph(f"Traffic condition: {escape(_pdf_safe(traffic_light.get('label', 'YELLOW - INSUFFICIENT DATA')))}", styles["BodyText"]),
        Paragraph(f"Severity: {escape(str(decision.get('severity', 'not available')))}", styles["BodyText"]),
        Paragraph(f"Recommendation: {escape(str(decision.get('recommended_action', 'not available')))}", styles["BodyText"]),
        Paragraph(f"Rule: {escape(str(decision.get('triggered_rule', 'not available')))}", styles["BodyText"]),
        Paragraph(f"Explanation: {escape(_pdf_safe(decision.get('reason', 'No explanation recorded.')))}", styles["BodyText"]),
        Paragraph(f"Risk score: {escape('Not scored (path safety override)' if decision.get('risk_score') is None and decision.get('triggered_rule') in {'person_in_free_left_path', 'pedestrian_vehicle_conflict', 'blocked_turn_path'} else str(decision.get('risk_score', 'not available')))} / 100 | Confidence: {escape(str(decision.get('confidence', 'not available')))}", styles["BodyText"]),
        Spacer(1, 8),
    ])
    joining_flow = (decision.get("input_metrics") or {}).get("joining_road_flow") or {}
    if joining_flow:
        story.append(Paragraph("Right-side joining-road flow", styles["Heading3"]))
        story.append(Paragraph(
            f"Flow status: {escape(str(joining_flow.get('status', 'uncertain')))}; moving observations: {escape(str(joining_flow.get('moving_vehicle_observations', 'not available')))}; stationary observations: {escape(str(joining_flow.get('stationary_vehicle_observations', 'not available')))}; stopped share: {escape(str(joining_flow.get('stationary_observation_ratio', 'not available')))}; peak stationary vehicles: {escape(str(joining_flow.get('maximum_stationary_vehicles', 'not available')))}; longest stationary duration: {escape(str(joining_flow.get('longest_stationary_duration_seconds', 'not available')))} seconds.",
            styles["BodyText"],
        ))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Signal update timing and color changes", styles["Heading2"]))
    timing = session.get("decision_timing") or {}
    timeline = session.get("decision_timeline") or []
    if timing.get("update_count"):
        story.append(Paragraph(
            f"Decision recalculations recorded: {escape(str(timing.get('update_count')))}; color transitions after the first displayed state: {escape(str(timing.get('signal_change_count', 0)))}; first advisory after job start: {_display(timing.get('first_update_after_seconds'), ' s')}; average update interval: {_display(timing.get('mean_update_interval_seconds'), ' s')}; shortest/longest interval: {_display(timing.get('minimum_update_interval_seconds'), ' / ')}{_display(timing.get('maximum_update_interval_seconds'), ' s')}.",
            styles["BodyText"],
        ))
        signal_rows = [["Elapsed job time (s)", "Signal", "Rule", "Reason"]]
        for event in timeline:
            signal_rows.append([
                _display(event.get("elapsed_seconds")),
                _pdf_safe(event.get("label", event.get("color", "unknown"))),
                Paragraph(escape(str(event.get("triggered_rule", "not recorded"))), styles["SmallNote"]),
                Paragraph(escape(_pdf_safe(event.get("reason", "No explanation recorded."))), styles["SmallNote"]),
            ])
        signal_table = Table(signal_rows, repeatRows=1, colWidths=[30 * mm, 42 * mm, 38 * mm, 60 * mm])
        signal_table.setStyle(_simple_table_style(colors))
        story.extend([signal_table, Spacer(1, 5)])
        if timing.get("event_history_truncated"):
            story.append(Paragraph("The event list reached its storage limit. The total transition counter remains recorded, but the table contains only the earliest saved events.", styles["SmallNote"]))
    else:
        story.append(Paragraph(
            "Signal transition timing was not captured for this analysis session. The saved data contains the final advisory only, so the number and exact times of earlier color changes cannot be reconstructed. New analysis sessions record update intervals and each color transition.",
            styles["SmallNote"],
        ))
    story.append(Spacer(1, 8))

    blockage_rows = [["Camera", "Start (clip s)", "End (clip s)", "Duration (s)", "Severity"]]
    for camera in session.get("camera_results", []):
        for event in camera.get("blockage_events") or []:
            blockage_rows.append([str(camera.get("camera_id", "unknown")), _display(event.get("started_at_video_seconds")), _display(event.get("ended_at_video_seconds")), _display(event.get("duration_seconds")), str(event.get("severity", "unknown"))])
    if len(blockage_rows) == 1 and any(camera.get("roi_status") == "not_configured" for camera in session.get("camera_results", [])):
        blockage_rows.append(["ROI not set", "", "", "", ""])
    if len(blockage_rows) == 1:
        blockage_rows.append(["No blockage event recorded", "-", "-", "-", "-"])
    story.append(PageBreak())
    story.append(Paragraph("Blockage events", styles["Heading2"]))
    blockage_table = Table(blockage_rows, repeatRows=1, colWidths=[32 * mm, 33 * mm, 33 * mm, 30 * mm, 28 * mm])
    blockage_table.setStyle(_simple_table_style(colors))
    story.extend([blockage_table, Spacer(1, 8)])

    story.append(Paragraph("Pedestrian activity and data limitations", styles["Heading2"]))
    pedestrian_rows = [["Camera", "All person tracks", "Route frame samples", "In route at end", "Person/vehicle conflict frames", "Vehicles in nearby buffer"]]
    for camera in session.get("camera_results", []):
        conflict_samples = camera.get("pedestrian_vehicle_overlap_samples")
        if conflict_samples is None and camera.get("roi_status") == "not_configured":
            conflict_samples = "not available (conflict ROI not configured)"
        elif conflict_samples is None:
            conflict_samples = "not available"
        zone_tracks = camera.get("zone_vehicle_tracks", {})
        nearby_vehicles = zone_tracks.get("vehicle_near_crossing_buffer", "not configured")
        route_now = camera.get("turn_path_pedestrian_now", camera.get("pedestrian_in_free_left_route_now"))
        route_count = camera.get("turn_path_pedestrian_count_now", camera.get("pedestrian_in_free_left_route_count_now", 0))
        route_text = f"yes ({route_count})" if route_now else "no"
        pedestrian_rows.append([
            _pdf_safe(camera.get("display_name", camera.get("camera_id", "camera"))),
            _display(camera.get("pedestrian_tracks")),
            _display(camera.get("pedestrian_in_free_left_samples")),
            route_text,
            str(conflict_samples),
            str(nearby_vehicles),
        ])
    pedestrian_table = Table(pedestrian_rows, repeatRows=1, colWidths=[35 * mm, 24 * mm, 25 * mm, 24 * mm, 37 * mm, 25 * mm])
    pedestrian_table.setStyle(_simple_table_style(colors))
    story.append(pedestrian_table)
    story.append(Paragraph(
        "Person tracks outside the checked camera 2 area are informational when a turn_path polygon is configured. With no polygon, the full camera 2 frame is checked. A person in the checked area triggers the pedestrian safety hold; a vehicle must remain stopped there for the configured interval.",
        styles["SmallNote"],
    ))
    for note in session.get("limitations", []):
        story.append(Paragraph(f"- {escape(str(note))}", styles["SmallNote"]))
    story.append(Paragraph("Estimated speed in km/h is omitted because camera calibration was not supplied. This advisory is for human review only, is not a safe phase-duration recommendation, and is not certified for direct signal control.", styles["SmallNote"]))

    document.build(story)
    return output_path


def _display(value: Any, suffix: str = "") -> str:
    return "-" if value is None else f"{value}{suffix}"


def _pdf_safe(value: Any) -> str:
    """Replace typography not supported by ReportLab's built-in Helvetica font."""

    text = str(value)
    for source, target in {"·": " - ", "–": "-", "—": "-", "→": "->", "×": "x", "“": '"', "”": '"', "’": "'"}.items():
        text = text.replace(source, target)
    return text.encode("ascii", errors="replace").decode("ascii")


def _simple_table_style(colors: Any) -> Any:
    from reportlab.platypus import TableStyle

    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c2cc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
