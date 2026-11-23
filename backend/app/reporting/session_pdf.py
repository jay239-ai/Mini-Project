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
        Paragraph("This report summarizes recorded analysis output. Camera-local track IDs do not establish global identity; camera counts may include the same vehicle in more than one view.", styles["SmallNote"]),
        Spacer(1, 10),
        Paragraph("Camera results", styles["Heading2"]),
    ]
    rows = [["Camera", "Frames", "Coverage", "Vehicle tracks*", "Queue area", "Max queue", "Avg wait", "Blocked s", "Blockage %"]]
    categories: dict[str, int] = {}
    total_camera_vehicle_tracks = 0
    for camera in session.get("camera_results", []):
        camera_categories = camera.get("vehicle_category_tracks", {})
        vehicle_tracks = sum(int(value) for value in camera_categories.values())
        zone_tracks = camera.get("zone_vehicle_tracks", {})
        approach_roi_tracks = zone_tracks.get("queue_area", "—")
        total_camera_vehicle_tracks += vehicle_tracks
        for name, count in camera_categories.items():
            categories[name] = categories.get(name, 0) + int(count)
        rows.append([
            str(camera.get("camera_id", "unknown")),
            _display(camera.get("frame_count_processed")),
            _display(camera.get("frame_coverage_percent"), "%"),
            str(vehicle_tracks),
            _pdf_safe(approach_roi_tracks),
            _display(camera.get("maximum_queue_vehicles")),
            _display(camera.get("average_wait_seconds"), " s"),
            _display(camera.get("total_blocked_seconds"), " s"),
            _display(camera.get("blockage_ratio_percent"), "%"),
        ])
    camera_table = Table(rows, repeatRows=1, colWidths=[19 * mm, 13 * mm, 16 * mm, 21 * mm, 17 * mm, 14 * mm, 17 * mm, 17 * mm, 17 * mm])
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
    story.extend([camera_table, Spacer(1, 7), Paragraph(f"* Vehicle tracks observed by cameras, not deduplicated junction-wide. Combined camera observations: {total_camera_vehicle_tracks}.", styles["SmallNote"]), Spacer(1, 8)])

    story.append(Paragraph("Vehicle category distribution", styles["Heading2"]))
    category_rows = [["Category", "Camera-local tracks"]] + [[name, str(count)] for name, count in sorted(categories.items())]
    if not categories:
        category_rows.append(["No category measurements available", "-"])
    category_table = Table(category_rows, colWidths=[60 * mm, 45 * mm])
    category_table.setStyle(_simple_table_style(colors))
    story.extend([category_table, Spacer(1, 8)])

    story.append(Paragraph("Vehicle observations by camera region", styles["Heading2"]))
    zone_rows = [["Camera", "Vehicle tracks", "Tracks observed in configured zones"]]
    for camera in session.get("camera_results", []):
        zone_tracks = camera.get("zone_vehicle_tracks", {})
        zone_text = "; ".join(f"{name}: {count}" for name, count in sorted(zone_tracks.items())) or "No region counts recorded"
        zone_rows.append([
            _pdf_safe(camera.get("display_name", camera.get("camera_id", "unknown"))),
            str(camera.get("vehicle_track_count", sum(int(value) for value in camera.get("vehicle_category_tracks", {}).values()))),
            Paragraph(escape(zone_text), styles["SmallNote"]),
        ])
    zone_table = Table(zone_rows, repeatRows=1, colWidths=[48 * mm, 28 * mm, 94 * mm])
    zone_table.setStyle(_simple_table_style(colors))
    identity = session.get("cross_camera_identity", {})
    estimated_unique = identity.get("estimated_unique_vehicles", identity.get("unique_vehicle_count_estimate_after_confirmed_merges"))
    upper_bound = identity.get("unique_vehicle_count_upper_bound")
    story.extend([
        zone_table,
        Spacer(1, 4),
        Paragraph(
            f"Estimated junction-wide unique vehicles after conservative cross-camera matching: {_display(estimated_unique)}; observed-track upper bound: {_display(upper_bound)}. Zone values are camera-local tracks seen in each polygon, not a calibrated line-crossing count; the same physical vehicle may appear in multiple cameras.",
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
        Paragraph(f"Risk score: {escape('Not scored (pedestrian safety override)' if decision.get('risk_score') is None and decision.get('triggered_rule') in {'person_in_free_left_path', 'pedestrian_vehicle_conflict'} else str(decision.get('risk_score', 'not available')))} / 100 | Confidence: {escape(str(decision.get('confidence', 'not available')))}", styles["BodyText"]),
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
        route_now = camera.get("pedestrian_in_free_left_route_now")
        route_count = camera.get("pedestrian_in_free_left_route_count_now", 0)
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
        "Person tracks anywhere in a frame are informational. Only a person whose footpoint enters a configured free-left or pedestrian-conflict polygon triggers the route safety rule. A person in the crossing-conflict polygon and a vehicle in that polygon or the optional nearby buffer in the same frame count as a conflict sample. Calibrate both polygons to the real camera view.",
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
