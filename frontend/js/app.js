(() => {
  const $ = (selector) => document.querySelector(selector);
  const cameraViews = [
    { key: 'approach', frame: 'trackingFrame', video: 'videoPreview', row: 'cameraRowApproach', status: 'approachCameraStatus', observations: 'approachObservations' },
    { key: 'turning', frame: 'turningTrackingFrame', video: 'pedestrianPreview', row: 'cameraRowTurning', status: 'turningCameraStatus', observations: 'turningObservations' },
    { key: 'exit', frame: 'exitTrackingFrame', video: 'oppositePreview', row: 'cameraRowExit', status: 'exitCameraStatus', observations: 'exitObservations' }
  ];
  const inputs = ['videoUpload', 'pedestrianUpload', 'oppositeUpload'];
  const fileLabels = ['videoFileName', 'pedestrianFileName', 'oppositeFileName'];
  const defaultSiteLabel = 'Junction A · North';
  const fileUrl = { videoPreview: null, pedestrianPreview: null, oppositePreview: null };
  const trackingFrameUrls = { approach: null, turning: null, exit: null };
  let activeLiveJobId = null;
  let activeAnalysisJobId = null;
  let activeUploadRequest = null;
  let stopRequested = false;
  let controlsLocked = false;
  let toastTimer;

  function selectedClipCount() {
    return inputs.filter((id) => Boolean(document.getElementById(id)?.files?.[0])).length;
  }

  function updateUploadState() {
    const count = selectedClipCount();
    const button = $('#analyzeBtn');
    button.textContent = count === 3 ? 'Analyze three clips' : `Analyze three clips (${count}/3)`;
    button.disabled = controlsLocked || count !== 3 || $('#sourceMode').value !== 'recorded';
    if ($('#sourceMode').value !== 'recorded') return;
    $('#analysisStatus').textContent = count === 3
      ? 'All 3 clips are ready. Click “Analyze three clips” to start detection.'
      : `${count} of 3 camera clips selected. Choose a clip for each camera; analysis starts when you click the button.`;
    $('#sourceStatus').textContent = count === 3 ? '3 CLIPS READY' : count ? `${count} OF 3 CLIPS READY` : 'WAITING FOR INPUT';
  }

  function setControlsLocked(locked) {
    controlsLocked = locked;
    [...inputs, 'reportUpload', 'sourceMode', 'performanceProfile', 'siteLabelInput', 'liveDeviceApproach', 'liveDeviceTurning', 'liveDeviceExit', 'liveUrlApproach', 'liveUrlTurning', 'liveUrlExit'].forEach((id) => {
      const element = document.getElementById(id);
      if (element) element.disabled = locked;
    });
    ['.upload-label', '.report-upload'].forEach((selector) => document.querySelectorAll(selector).forEach((label) => {
      label.classList.toggle('disabled', locked);
      label.setAttribute('aria-disabled', String(locked));
    }));
    $('#refreshUsbCameras').disabled = locked;
    $('#analyzeBtn').disabled = locked || selectedClipCount() !== 3 || $('#sourceMode').value !== 'recorded';
    $('#startLiveBtn').disabled = locked;
  }

  function notify(message) {
    $('#toast').textContent = message;
    $('#toast').classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => $('#toast').classList.remove('show'), 3500);
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  }

  async function apiJson(url, options = {}) {
    const { timeoutMs = 10000, ...fetchOptions } = options;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { cache: 'no-store', ...fetchOptions, signal: controller.signal });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || `Request failed (${response.status}).`);
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error(`The local service did not respond in time (${url}).`);
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function refreshApiStatus() {
    const badge = $('#apiStatus');
    try {
      await apiJson('/api/v1/health/live', { timeoutMs: 2500 });
      badge.innerHTML = '<i></i> API CONNECTED';
      badge.title = 'The local analysis service is responding.';
    } catch {
      badge.innerHTML = '<i></i> API OFFLINE';
      badge.title = 'Start the local analysis server to use the dashboard.';
    }
  }

  function selectedSite() {
    return ($('#siteLabelInput')?.value || defaultSiteLabel).trim();
  }

  function updateSiteLabel() {
    $('#sidebarSite').textContent = selectedSite() || 'Site not set';
  }

  function renderHistory(history, filename = null) {
    const reports = Number(history.report_count || 0);
    const level = history.peak_congestion_level || 'none';
    const risk = level === 'none' ? 'NO REPORTS' : String(level).toUpperCase();
    const riskClass = ['high', 'severe', 'very high'].includes(String(level).toLowerCase()) ? 'high' : ['medium', 'moderate'].includes(String(level).toLowerCase()) ? 'medium' : 'low';
    $('#historyRisk').textContent = risk;
    $('#historyRisk').className = `risk-pill ${riskClass}`;
    $('#reportName').textContent = filename || (reports ? `${reports} report${reports === 1 ? '' : 's'} stored for ${history.site_label}` : 'No report uploaded for this site');
    const period = history.report_period_start ? ` · period ${history.report_period_start}${history.report_period_end && history.report_period_end !== history.report_period_start ? ` to ${history.report_period_end}` : ''}` : '';
    $('#reportSummary').textContent = reports ? `Historical context only · ${history.confidence || 'unknown'} extraction confidence${period}` : 'Import a text-based traffic report PDF to add historical context for this site.';
    $('#reportCount').textContent = reports;
    $('#incidentCount').textContent = history.incident_count ?? '—';
    $('#violationCount').textContent = history.violation_count ?? '—';
    $('#peakCongestion').textContent = history.peak_congestion_level || '—';
    $('#historicalMaxQueue').textContent = history.maximum_queue_vehicles == null ? '—' : `${history.maximum_queue_vehicles} vehicles`;
    $('#historicalMaxQueueLength').textContent = history.maximum_queue_length_meters == null ? '—' : `${Number(history.maximum_queue_length_meters).toFixed(1)} m`;
    $('#averageHistoricalDelay').textContent = history.average_delay_minutes == null ? '—' : `${Number(history.average_delay_minutes).toFixed(1)} min`;
    $('#longestHistoricalDelay').textContent = history.longest_delay_minutes == null ? '—' : `${Number(history.longest_delay_minutes).toFixed(1)} min`;
    $('#historyReportRows').innerHTML = (history.reports || []).map((report) => {
      const period = report.report_period_start ? `${report.report_period_start}${report.report_period_end && report.report_period_end !== report.report_period_start ? ` – ${report.report_period_end}` : ''}` : 'Date not recognized';
      return `<article><strong>${escapeHtml(period)}</strong><span>${report.incident_count ?? '—'} jams · queue ${report.maximum_queue_vehicles ?? '—'} vehicles${report.maximum_queue_length_meters == null ? '' : ` / ${Number(report.maximum_queue_length_meters).toFixed(1)} m`}</span><small>Delay ${report.average_delay_minutes == null ? '—' : `${Number(report.average_delay_minutes).toFixed(1)} min avg`} · extraction ${escapeHtml(report.extraction_confidence || 'unknown')}</small></article>`;
    }).join('');
    const warnings = Array.isArray(history.warnings) ? history.warnings : [];
    $('#historyWarnings').innerHTML = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('');
    $('#historyWarnings').hidden = warnings.length === 0;
  }

  async function loadHistory() {
    const site = selectedSite();
    if (!site) return;
    updateSiteLabel();
    try {
      const query = new URLSearchParams({ site_label: site });
      const history = await apiJson(`/api/v1/history/context?${query}`);
      renderHistory(history);
      $('#historyStatus').textContent = history.raw_pdf_retained === false
        ? 'PDFs are processed locally. Only extracted summary numbers are stored in SQLite; the PDF and full text are discarded.'
        : 'Historical context is read from the local database.';
    } catch (error) {
      $('#historyStatus').textContent = error.message;
    }
  }

  async function importReport(file) {
    if (!file) return;
    if (controlsLocked) return;
    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      notify('Choose a PDF traffic report.');
      return;
    }
    if (!selectedSite()) {
      notify('Enter the site name before importing its report.');
      return;
    }
    const button = document.querySelector('.report-upload');
    setControlsLocked(true);
    button.textContent = 'Reading report…';
    button.classList.add('disabled');
    $('#historyStatus').textContent = 'Extracting supported traffic facts locally…';
    const payload = new FormData();
    payload.append('site_label', selectedSite());
    payload.append('file', file);
    try {
      const result = await apiJson('/api/v1/history/reports', { method: 'POST', body: payload });
      const summary = await apiJson(`/api/v1/history/context?${new URLSearchParams({ site_label: selectedSite() })}`);
      renderHistory(summary, file.name);
      $('#historyStatus').textContent = result.status === 'duplicate'
        ? 'This exact report is already stored for this site; it was not counted twice.'
        : 'Report imported. The original PDF and extracted text were discarded; only supported numeric summaries were saved locally.';
      if (result.extraction_warnings?.length) {
        $('#historyWarnings').innerHTML = result.extraction_warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('');
        $('#historyWarnings').hidden = false;
      }
      notify(result.status === 'duplicate' ? 'Report already imported.' : 'Historical report imported.');
    } catch (error) {
      $('#historyStatus').textContent = error.message;
      notify(error.message);
    } finally {
      button.textContent = 'Import traffic report PDF';
      button.classList.remove('disabled');
      $('#reportUpload').value = '';
      setControlsLocked(false);
    }
  }

  function showLocalVideo(file, cameraIndex) {
    const videoId = cameraViews[cameraIndex].video;
    const video = document.getElementById(videoId);
    const trackedImage = document.getElementById(cameraViews[cameraIndex].frame);
    const empty = trackedImage.closest('.tracking-image-wrap')?.querySelector('.tracking-empty');
    if (!file) {
      if (fileUrl[videoId]) URL.revokeObjectURL(fileUrl[videoId]);
      fileUrl[videoId] = null;
      video.pause();
      video.removeAttribute('src');
      video.load();
      video.hidden = true;
      trackedImage.hidden = true;
      trackedImage.removeAttribute('src');
      empty?.removeAttribute('hidden');
      if (empty) empty.textContent = 'Choose a clip or connect a live camera';
      document.getElementById(fileLabels[cameraIndex]).textContent = 'No clip selected';
      document.getElementById(cameraViews[cameraIndex].status).textContent = 'Waiting';
      document.getElementById(cameraViews[cameraIndex].observations).textContent = 'Vehicle and pedestrian counts appear while frames are analyzed.';
      if (cameraIndex === 2) $('#joiningFlowStatus').textContent = 'Joining-road movement: waiting for camera data.';
      updateUploadState();
      return;
    }
    if (fileUrl[videoId]) URL.revokeObjectURL(fileUrl[videoId]);
    fileUrl[videoId] = URL.createObjectURL(file);
    video.src = fileUrl[videoId];
    video.hidden = false;
    video.muted = true;
    trackedImage.hidden = true;
    trackedImage.removeAttribute('src');
    if (empty) {
      empty.textContent = 'Preview ready · click Analyze three clips to show tracking';
      empty.setAttribute('hidden', '');
    }
    document.getElementById(cameraViews[cameraIndex].row).classList.remove('active');
    document.getElementById(fileLabels[cameraIndex]).textContent = file.name;
    document.getElementById(cameraViews[cameraIndex].status).textContent = 'Clip selected';
    document.getElementById(cameraViews[cameraIndex].observations).textContent = 'Clip ready · counts appear as soon as analysis starts.';
    if (cameraIndex === 2) $('#joiningFlowStatus').textContent = 'Joining-road movement: click Analyze to classify moving and stopped vehicles.';
    $('#downloadReportButton').hidden = true;
    renderDecision({ severity: 'insufficient', confidence: 'low', reason: 'New camera clips selected. Run analysis to calculate the current traffic condition.' });
    updateUploadState();
  }

  const pause = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

  function resetTrackingViews() {
    cameraViews.forEach((camera) => {
      const image = document.getElementById(camera.frame);
      image.hidden = true;
      image.removeAttribute('src');
      image.closest('.tracking-image-wrap')?.querySelector('.tracking-empty')?.removeAttribute('hidden');
      document.getElementById(camera.row).classList.remove('active');
      document.getElementById(camera.observations).textContent = 'Waiting for analyzed frames…';
    });
    $('#joiningFlowStatus').textContent = 'Joining-road movement: waiting for camera data.';
  }

  function uploadClips(payload) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      activeUploadRequest = request;
      request.open('POST', '/api/v1/analysis/recorded');
      request.upload.addEventListener('progress', (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round(event.loaded / event.total * 100);
        $('#progressPercent').textContent = `${percent}%`;
        $('#progressLabel').textContent = 'Uploading clips';
        $('#analysisStatus').textContent = `Uploading camera clips · ${percent}%`;
      });
      request.addEventListener('load', () => {
        let data = {};
        try { data = JSON.parse(request.responseText || '{}'); } catch { /* show generic request error */ }
        if (request.status < 200 || request.status >= 300) reject(new Error(data.detail || 'Could not upload clips.'));
        else resolve(data);
      });
      request.addEventListener('error', () => reject(new Error('Upload failed. Check the local server and try again.')));
      request.addEventListener('abort', () => reject(new Error('Upload canceled.')));
      request.send(payload);
    });
  }

  async function updateJobViews(job, lastFrameSequences) {
    $('#progressPercent').textContent = `${job.progress_percent || 0}%`;
    await Promise.all(cameraViews.map(async (camera) => {
      const counts = job.live_counts_by_camera?.[camera.key];
      const progress = job.camera_progress?.[camera.key] || 0;
      const cameraState = job.camera_states?.[camera.key];
      const fps = counts?.inference_fps;
      const sequence = job.frame_sequences?.[camera.key] || 0;
      const cameraError = job.camera_errors?.[camera.key];
      const cameraWarnings = job.camera_warnings?.[camera.key] || [];
      const speed = fps != null
        ? (counts.frame_age_ms != null
          ? `${Number(fps).toFixed(1)} / 30 FPS · frame ${counts.frame_age_ms} ms old · dropped ${counts.dropped_capture_frames ?? 0}`
          : `${Number(fps).toFixed(1)} / 30 FPS · processed ${counts.processing_fps ?? '—'} FPS · ${counts.realtime_factor ?? '—'}× clip speed`)
        : cameraState === 'error' ? 'Camera error' : cameraState || `${progress}% complete`;
      $(`#${camera.status}`).textContent = cameraError
        ? `Clip error · ${cameraError}`
        : cameraWarnings.length ? `Clip note · ${cameraWarnings[0]}` : speed;
      const observationElement = document.getElementById(camera.observations);
      if (sequence > 0 && counts?.inference_frames > 0) {
        const routePeople = Number(counts.pedestrians_in_free_left_route || 0);
        if (camera.key === 'approach') {
          observationElement.textContent = `Main-lane vehicles now: ${counts.vehicles_in_frame ?? 0} · main-lane tracks: ${counts.approach_lane_vehicle_tracks ?? 0} · free-left now: ${counts.free_left_vehicles_in_frame ?? 0} · free-left tracks: ${counts.free_left_vehicle_tracks ?? 0} · queue span: ${Number(counts.free_left_queue_length_pixels ?? 0).toFixed(0)} px · stopped delay: ${Number(counts.free_left_stopped_delay_seconds ?? 0).toFixed(1)} s`;
        } else if (camera.key === 'turning') {
          observationElement.textContent = `Turn path: ${counts.pedestrians_in_frame ?? 0} pedestrians · ${counts.turn_path_stationary_vehicle_count ?? 0} stopped vehicles · ${counts.turn_path_objects_in_frame ?? 0} total objects`;
        } else {
          observationElement.textContent = `Vehicles in frame: ${counts.vehicles_in_frame ?? 0} · vehicle tracks: ${counts.vehicle_tracks ?? 0} · People in monitored path: ${routePeople}`;
        }
      } else if (cameraState === 'error') observationElement.textContent = cameraError ? `This clip could not be analyzed: ${cameraError}` : 'Camera unavailable · check the source address or device connection.';
      else if (cameraWarnings.length) observationElement.textContent = `Video input note: ${cameraWarnings.join(' ')}`;
      else if (sequence > 0) observationElement.textContent = 'Camera preview connected · loading the tracking model…';
      else observationElement.textContent = 'Waiting for the camera to connect…';
      if (camera.key === 'exit') {
        const flowStatus = $('#joiningFlowStatus');
        if (counts) {
          const vehicles = Number(counts.joining_road_vehicles || 0);
          const moving = Number(counts.joining_road_moving_vehicles || 0);
          const stationary = Number(counts.joining_road_stationary_vehicles || 0);
          const unclassified = Number(counts.joining_road_unclassified_vehicles || 0);
          flowStatus.textContent = vehicles === 0
            ? 'Joining-road movement: clear in this frame.'
            : `Joining-road movement: ${moving} moving · ${stationary} stopped${unclassified ? ` · ${unclassified} not yet classified` : ''}.`;
        } else {
          flowStatus.textContent = `Joining-road movement: ${cameraState === 'error' ? 'unavailable · check camera' : 'waiting for camera evidence'}.`;
        }
      }
      $(`#${camera.row}`).classList.toggle('active', ['processing', 'streaming'].includes(job.status) && cameraState === 'analyzing');
      if (sequence <= lastFrameSequences[camera.key]) return;
      try {
        const response = await fetch(`/api/v1/analysis/${job.job_id}/frames/${camera.key}?sequence=${sequence}`, { cache: 'no-store' });
        if (!response.ok) return;
        if (trackingFrameUrls[camera.key]) URL.revokeObjectURL(trackingFrameUrls[camera.key]);
        trackingFrameUrls[camera.key] = URL.createObjectURL(await response.blob());
        const image = document.getElementById(camera.frame);
        image.src = trackingFrameUrls[camera.key];
        image.hidden = false;
        image.closest('.tracking-image-wrap')?.querySelector('.tracking-empty')?.setAttribute('hidden', '');
        const video = document.getElementById(camera.video);
        video.pause();
        video.hidden = true;
        lastFrameSequences[camera.key] = sequence;
      } catch { /* next poll retries the newest frame */ }
    }));
  }

  function renderDecision(decision = {}) {
    const severity = String(decision.severity || 'insufficient').toLowerCase();
    const fallbackColor = severity === 'high' ? 'red' : severity === 'low' && decision.confidence === 'moderate' ? 'green' : 'yellow';
    const color = decision.traffic_light?.color || fallbackColor;
    const label = decision.traffic_light?.label || (color === 'green' ? 'GREEN · FLOWING' : color === 'red' ? 'RED · HIGH PRESSURE' : severity === 'insufficient' ? 'YELLOW · INSUFFICIENT DATA' : 'YELLOW · CAUTION');
    const lamp = $('#trafficLamp');
    lamp.dataset.state = color;
    lamp.setAttribute('aria-label', `Traffic condition advisory: ${label}`);
    $('#trafficStateLabel').textContent = label;
    $('#decisionReason').textContent = decision.reason || 'Waiting for current camera evidence.';
    const evidence = decision.traffic_light?.evidence_quality || (decision.confidence === 'moderate' ? 'Evidence quality: sufficient observation across the required camera views.' : 'Evidence quality: limited. Yellow means wait for more reliable camera evidence.');
    $('#decisionEvidence').textContent = `${evidence} Advisory only; this does not control a real traffic signal.`;
  }

  function renderAnalysis(jobId, result = {}) {
    renderDecision(result.junction_decision || {});
    const reportLink = $('#downloadReportButton');
    if (result.report_url) {
      reportLink.dataset.reportUrl = result.report_url;
      reportLink.dataset.filename = `traffic-analysis-${jobId.slice(0, 12)}.pdf`;
      reportLink.hidden = false;
    } else {
      delete reportLink.dataset.reportUrl;
      delete reportLink.dataset.filename;
      reportLink.hidden = true;
    }
    if (result.historical_context) renderHistory(result.historical_context);
    $('#sourceStatus').textContent = result.data_status === 'complete' ? 'ANALYSIS COMPLETE' : 'PARTIAL ANALYSIS';
  }

  async function runRecordedAnalysis() {
    const files = inputs.map((id) => document.getElementById(id).files[0]);
    if (files.some((file) => !file)) { notify('Choose one clip for each camera.'); return; }
    if (!selectedSite()) { notify('Enter the intersection/site name first.'); return; }
    stopRequested = false;
    activeAnalysisJobId = null;
    $('#stopAnalysisBtn').hidden = false;
    $('#stopAnalysisBtn').disabled = false;
    setControlsLocked(true);
    resetTrackingViews();
    $('#downloadReportButton').hidden = true;
    $('#analysisProgress').hidden = false;
    renderDecision({ severity: 'insufficient', confidence: 'low', reason: 'Collecting up to two seconds of initial camera evidence; person-in-path alerts are immediate.' });
    $('#analysisStatus').textContent = 'Uploading clips now · analysis starts automatically when they arrive';
    $('#analyzeBtn').textContent = 'Starting analysis…';
    $('#progressLabel').textContent = 'Uploading clips and starting analysis';
    $('#progressPercent').textContent = '0%';
    try {
      // Start the local previews and clip upload immediately; avoid a separate
      // runtime-probe request on the click-to-analysis critical path.
      cameraViews.forEach((camera) => {
        const video = document.getElementById(camera.video);
        if (!video.hidden) { video.currentTime = 0; video.play().catch(() => {}); }
      });
      if (stopRequested) return;
      const payload = new FormData();
      ['camera1', 'camera2', 'camera3'].forEach((name, index) => payload.append(name, files[index]));
      payload.append('performance_profile', $('#performanceProfile').value);
      payload.append('site_label', selectedSite());
      const started = await uploadClips(payload);
      activeUploadRequest = null;
      activeAnalysisJobId = started.job_id;
      if (stopRequested) await apiJson(`/api/v1/analysis/${activeAnalysisJobId}/stop`, { method: 'POST' });
      const lastFrameSequences = { approach: 0, turning: 0, exit: 0 };
      let job;
      do {
        await pause(200);
        job = await apiJson(`/api/v1/analysis/${started.job_id}`);
        $('#progressLabel').textContent = job.stop_requested ? 'Stopping after current frame…' : job.message || 'Analyzing camera views';
        $('#analysisStatus').textContent = job.stop_requested || stopRequested ? 'Stopping analysis safely…' : `${job.message || 'Analyzing'}${job.compute_device ? ` · ${job.compute_device}` : ''}`;
        renderDecision(job.decision || {});
        await updateJobViews(job, lastFrameSequences);
      } while (['queued', 'initializing', 'processing', 'stopping'].includes(job.status));
      if (job.status === 'stopped') {
        if (job.result) renderAnalysis(started.job_id, job.result);
        $('#analysisStatus').textContent = job.result?.report_error
          ? `Analysis stopped, but the PDF report could not be created: ${job.result.report_error}`
          : job.result?.report_url ? 'Analysis stopped. The PDF report is ready; select Download analysis report when you want it.'
            : 'Analysis stopped before usable frames were available, so no report is ready.';
        return;
      }
      if (!['complete', 'partial'].includes(job.status)) throw new Error(job.error || 'Analysis failed.');
      renderAnalysis(started.job_id, job.result);
      $('#analysisStatus').textContent = job.status === 'partial'
        ? 'Some camera clips could not be analyzed. The partial results and report are ready to review.'
        : 'All three camera clips analyzed.';
      await loadHistory();
    } catch (error) {
      if (stopRequested) $('#analysisStatus').textContent = 'Analysis stopped.';
      else {
        $('#analysisStatus').textContent = error.message;
        notify(error.message);
      }
    } finally {
      cameraViews.forEach((camera) => document.getElementById(camera.video).pause());
      activeUploadRequest = null;
      activeAnalysisJobId = null;
      $('#stopAnalysisBtn').hidden = true;
      $('#stopAnalysisBtn').disabled = false;
      setControlsLocked(false);
      $('#analyzeBtn').textContent = 'Analyze three clips';
      $('#analysisProgress').hidden = true;
    }
  }

  async function discoverUsbCameras() {
    const button = $('#refreshUsbCameras');
    button.disabled = true;
    $('#liveDeviceStatus').textContent = 'Checking connected USB camera devices…';
    try {
      const data = await apiJson('/api/v1/analysis/devices');
      const selects = [$('#liveDeviceApproach'), $('#liveDeviceTurning'), $('#liveDeviceExit')];
      const previousAssignments = selects.map((select) => select.value);
      selects.forEach((select) => {
        select.innerHTML = '<option value="">Not connected</option>';
        (data.devices || []).forEach((device) => {
          const option = document.createElement('option');
          option.value = String(device.device_index);
          option.dataset.status = device.status || 'ready';
          option.textContent = device.status === 'black_frame'
            ? `USB camera ${device.device_index} · black frames detected`
            : `USB camera ${device.device_index} · ${device.width}×${device.height}`;
          select.append(option);
        });
      });
      const assigned = new Set();
      selects.forEach((select, index) => {
        const previous = previousAssignments[index];
        if (previous && [...select.options].some((option) => option.value === previous) && !assigned.has(previous)) {
          select.value = previous;
          assigned.add(previous);
        }
      });
      selects.forEach((select) => {
        if (select.value) return;
        const available = [...select.options].find((option) => option.value && !assigned.has(option.value) && option.dataset.status !== 'black_frame');
        if (available) {
          select.value = available.value;
          assigned.add(available.value);
        }
      });
      const readyCount = (data.devices || []).filter((device) => device.status !== 'black_frame').length;
      const blackCount = (data.devices || []).length - readyCount;
      $('#liveDeviceStatus').textContent = data.devices?.length
        ? `Found ${readyCount} working camera(s)${blackCount ? ` and ${blackCount} device(s) returning black frames` : ''}. Working cameras were assigned first. For a black phone feed, enable USB webcam/UVC mode, close other camera apps, and scan again.`
        : 'No USB webcam is exposed to Windows. Enable the phone’s USB webcam/UVC mode, reconnect, and scan again; opening the phone Camera app alone is not enough. Or enter an RTSP/HTTP stream URL from a phone camera-streaming app.';
    } catch (error) {
      $('#liveDeviceStatus').textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  async function runLiveAnalysis() {
    const selected = [
      $('#liveUrlApproach').value.trim() || $('#liveDeviceApproach').value,
      $('#liveUrlTurning').value.trim() || $('#liveDeviceTurning').value,
      $('#liveUrlExit').value.trim() || $('#liveDeviceExit').value
    ];
    const devices = selected.filter((value) => value !== '');
    if (!devices.length) { $('#liveDeviceStatus').textContent = 'Find a USB camera or enter an RTSP/HTTP stream address.'; return; }
    if (new Set(devices).size !== devices.length) { $('#liveDeviceStatus').textContent = 'Choose a different device for each camera view.'; return; }
    if (!selectedSite()) { $('#liveDeviceStatus').textContent = 'Enter the intersection/site name first.'; return; }
    stopRequested = false;
    activeAnalysisJobId = null;
    $('#stopAnalysisBtn').hidden = false;
    $('#stopAnalysisBtn').disabled = false;
    setControlsLocked(true);
    resetTrackingViews();
    $('#analysisProgress').hidden = false;
    $('#progressLabel').textContent = 'Starting live streams';
    $('#downloadReportButton').hidden = true;
    renderDecision({ severity: 'insufficient', confidence: 'low', reason: 'Collecting up to two seconds of initial camera evidence; person-in-path alerts are immediate.' });
    try {
      if (stopRequested) return;
      const payload = new FormData();
      ['camera1', 'camera2', 'camera3'].forEach((name, index) => { if (selected[index] !== '') payload.append(name, selected[index]); });
      payload.append('site_label', selectedSite());
      payload.append('performance_profile', $('#performanceProfile').value);
      const started = await apiJson('/api/v1/analysis/live/start', { method: 'POST', body: payload });
      activeLiveJobId = started.job_id;
      activeAnalysisJobId = started.job_id;
      if (stopRequested) await apiJson(`/api/v1/analysis/${activeAnalysisJobId}/stop`, { method: 'POST' });
      $('#startLiveBtn').hidden = true;
      $('#analysisStatus').textContent = 'Connecting to live cameras…';
      const lastFrameSequences = { approach: 0, turning: 0, exit: 0 };
      let job;
      do {
        await pause(200);
        job = await apiJson(`/api/v1/analysis/${activeLiveJobId}`);
        $('#analysisProgress').hidden = job.status === 'streaming';
        $('#analysisStatus').textContent = job.stop_requested || stopRequested ? 'Stopping camera streams safely…' : job.message || 'Live camera analysis';
        $('#sourceStatus').textContent = job.status === 'streaming' ? 'LIVE CAMERAS' : job.status.toUpperCase();
        renderDecision(job.decision || {});
        await updateJobViews(job, lastFrameSequences);
      } while (['initializing', 'streaming', 'stopping'].includes(job.status));
      if (job.status === 'failed' && !job.result) throw new Error(job.error || job.message || 'Live camera analysis failed.');
      const liveResult = job.result || {
        camera_results: job.camera_results || [],
        cross_camera_identity: job.live_identity || {},
        historical_context: job.historical_context,
        junction_decision: job.junction_decision || job.decision || {},
        report_url: job.result?.report_url,
        session_mode: 'live_camera'
      };
      renderAnalysis(activeLiveJobId, liveResult);
      $('#analysisStatus').textContent = liveResult.report_error ? `Live analysis stopped, but the PDF report could not be created: ${liveResult.report_error}`
        : liveResult.report_url ? 'Live analysis stopped. The session PDF is ready; select Download analysis report when you want it.'
          : job.status === 'failed' ? `Live session ended with an input error. ${liveResult.report_url ? 'The partial PDF report is available to download.' : job.message || 'No report is available.'}`
            : job.message || 'Live camera session stopped.';
      await loadHistory();
    } catch (error) {
      if (stopRequested) $('#analysisStatus').textContent = 'Analysis stopped.';
      else {
        $('#analysisStatus').textContent = error.message;
        $('#liveDeviceStatus').textContent = error.message;
        notify(error.message);
      }
    } finally {
      activeLiveJobId = null;
      activeAnalysisJobId = null;
      $('#stopAnalysisBtn').hidden = true;
      $('#stopAnalysisBtn').disabled = false;
      $('#startLiveBtn').hidden = false;
      setControlsLocked(false);
      $('#analysisProgress').hidden = true;
    }
  }

  function applySourceMode() {
    const live = $('#sourceMode').value === 'live_usb';
    document.body.dataset.sourceMode = live ? 'live' : 'upload';
    $('#recordedSourceControls').hidden = live;
    $('#liveSourceControls').hidden = !live;
    $('#analyzeBtn').hidden = live;
    resetTrackingViews();
    cameraViews.forEach((camera) => {
      const video = document.getElementById(camera.video);
      video.pause();
      video.hidden = live || !fileUrl[camera.video];
    });
    $('#analysisStatus').textContent = live ? 'Select wired camera devices, then start live analysis.' : 'Choose one clip for each camera.';
    $('#sourceStatus').textContent = live ? 'LIVE CAMERA MODE' : 'UPLOAD MODE';
    renderDecision({ severity: 'insufficient', confidence: 'low', reason: live ? 'Waiting for all three live views to provide enough camera evidence.' : 'Choose three clips and analyze them to determine the traffic condition.' });
    updateUploadState();
  }
  $('#sourceMode').addEventListener('change', applySourceMode);
  applySourceMode();
  inputs.forEach((id, index) => document.getElementById(id).addEventListener('change', (event) => showLocalVideo(event.target.files[0] || null, index)));
  $('#reportUpload').addEventListener('change', (event) => importReport(event.target.files[0]));
  $('#downloadReportButton').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    const reportUrl = button.dataset.reportUrl;
    if (!reportUrl || button.disabled) return;
    button.disabled = true;
    try {
      const response = await fetch(reportUrl, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Report download failed (${response.status}).`);
      const blobUrl = URL.createObjectURL(await response.blob());
      const anchor = document.createElement('a');
      anchor.href = blobUrl;
      anchor.download = button.dataset.filename || 'traffic-analysis.pdf';
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(blobUrl), 30000);
    } catch (error) {
      notify(error.message);
    } finally {
      button.disabled = false;
    }
  });
  $('#siteLabelInput')?.addEventListener('change', loadHistory);
  $('#siteLabelInput')?.addEventListener('input', updateSiteLabel);
  $('#analyzeBtn').addEventListener('click', runRecordedAnalysis);
  $('#refreshUsbCameras').addEventListener('click', discoverUsbCameras);
  $('#startLiveBtn').addEventListener('click', runLiveAnalysis);
  $('#stopAnalysisBtn').addEventListener('click', async () => {
    stopRequested = true;
    $('#stopAnalysisBtn').disabled = true;
    $('#analysisStatus').textContent = 'Stopping analysis…';
    $('#progressLabel').textContent = 'Stopping analysis…';
    if (activeUploadRequest) {
      $('#analysisStatus').textContent = 'Finishing the video upload, then preparing a partial report…';
      $('#progressLabel').textContent = 'Finishing upload before stopping';
      return;
    }
    if (!activeAnalysisJobId) return;
    try { await apiJson(`/api/v1/analysis/${activeAnalysisJobId}/stop`, { method: 'POST' }); }
    catch (error) { $('#analysisStatus').textContent = `Stop request could not be confirmed: ${error.message}`; }
  });
  window.addEventListener('beforeunload', () => {
    if (activeAnalysisJobId) navigator.sendBeacon(`/api/v1/analysis/${activeAnalysisJobId}/stop`);
    Object.values(fileUrl).forEach((url) => { if (url) URL.revokeObjectURL(url); });
    Object.values(trackingFrameUrls).forEach((url) => { if (url) URL.revokeObjectURL(url); });
  });

  updateSiteLabel();
  refreshApiStatus();
  loadHistory();
  window.setInterval(refreshApiStatus, 30000);
})();
