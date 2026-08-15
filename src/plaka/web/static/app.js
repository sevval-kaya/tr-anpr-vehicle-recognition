"use strict";

// Re-scans the document for <i data-lucide="..."> placeholders and swaps
// each for an inline SVG — must be called again after any innerHTML
// update that introduces new data-lucide elements (Lucide only processes
// the DOM once per call, it doesn't watch for changes).
function refreshIcons() {
  if (window.lucide) lucide.createIcons();
}
refreshIcons();

// ---------- tabs ----------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab !== "camera") stopCamera();
  });
});

// Lucide doesn't have a dedicated "motorcycle" icon (verified against the
// CDN directly — that name 404s); "bike" is the closest match.
const VEHICLE_TYPE_INFO = {
  car: { label: "Otomobil", icon: "car" },
  truck: { label: "Kamyon", icon: "truck" },
  bus: { label: "Otobüs", icon: "bus" },
  motorcycle: { label: "Motosiklet", icon: "bike" },
};

function vehicleTypeLabel(type) {
  return (VEHICLE_TYPE_INFO[type] || { label: type }).label;
}

function vehicleTypeCellHtml(type) {
  const info = VEHICLE_TYPE_INFO[type] || { label: type, icon: "help-circle" };
  return `<span class="vehicle-type-cell"><i data-lucide="${info.icon}"></i>${info.label}</span>`;
}

function renderVehicleRows(tbody, vehicles) {
  tbody.innerHTML = "";
  if (!vehicles.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="muted">Araç tespit edilmedi</td></tr>';
    refreshIcons();
    return;
  }
  for (const v of vehicles) {
    const tr = document.createElement("tr");
    const plateCell = v.plate_text
      ? `<span class="plate-pill ${v.plate_valid ? "" : "invalid"}">${v.plate_text}</span>`
      : '<span class="muted">—</span>';
    tr.innerHTML = `<td>${vehicleTypeCellHtml(v.vehicle_type)}</td><td>${plateCell}</td>` +
      `<td>${Math.round(v.detection_confidence * 100)}%</td>`;
    tbody.appendChild(tr);
  }
  refreshIcons();
}

function setupDropzone(dropzoneId, inputId, submitId, onFileChosen) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  const submitBtn = document.getElementById(submitId);

  const choose = (file) => {
    if (!file) return;
    submitBtn.disabled = false;
    onFileChosen(file);
  };

  input.addEventListener("change", () => choose(input.files[0]));
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      choose(input.files[0]);
    }
  });
}

// ---------- photo ----------
let selectedPhotoFile = null;
setupDropzone("photo-dropzone", "photo-input", "photo-submit", (file) => {
  selectedPhotoFile = file;
  document.getElementById("photo-dropzone").querySelector(".dropzone-text").textContent = file.name;
});

document.getElementById("photo-submit").addEventListener("click", async () => {
  if (!selectedPhotoFile) return;
  const statusEl = document.getElementById("photo-status");
  const submitBtn = document.getElementById("photo-submit");
  submitBtn.disabled = true;
  statusEl.textContent = "Analiz ediliyor…";
  document.getElementById("photo-result").classList.add("hidden");

  try {
    const formData = new FormData();
    formData.append("file", selectedPhotoFile);
    const res = await fetch("/api/infer/image", { method: "POST", body: formData });
    if (!res.ok) throw new Error(`sunucu hatası (${res.status})`);
    const data = await res.json();

    document.getElementById("photo-result-image").src =
      `data:image/jpeg;base64,${data.annotated_image_base64}`;
    renderVehicleRows(
      document.querySelector("#photo-results-table tbody"),
      data.vehicles
    );
    document.getElementById("photo-result").classList.remove("hidden");
    statusEl.textContent = `${data.vehicles.length} araç tespit edildi.`;
  } catch (err) {
    statusEl.textContent = `Hata: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});

// ---------- video ----------
let selectedVideoFile = null;
let videoPollTimer = null;

// ---- rotation preview (client-side, before upload) ----
// Draws `sourceCanvas` into `destCanvas` rotated clockwise by `degrees`,
// matching cv2.rotate()'s convention exactly (see verify_rotation_preview.py
// / docs/decisions.md #39 — checked pixel-for-pixel against the backend's
// actual apply_rotation() so a thumbnail the user picks here really is
// what the server will produce) — center-translate-rotate is
// orientation-agnostic for any 90°-multiple, no per-case offset math needed.
function drawRotated(sourceCanvas, destCanvas, degrees) {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  if (degrees === 90 || degrees === 270) {
    destCanvas.width = h;
    destCanvas.height = w;
  } else {
    destCanvas.width = w;
    destCanvas.height = h;
  }
  const ctx = destCanvas.getContext("2d");
  ctx.save();
  ctx.translate(destCanvas.width / 2, destCanvas.height / 2);
  ctx.rotate((degrees * Math.PI) / 180);
  ctx.drawImage(sourceCanvas, -w / 2, -h / 2);
  ctx.restore();
}

const ROTATION_VALUES = [0, 90, 180, 270];
let selectedRotation = 0;

function setSelectedRotation(value) {
  selectedRotation = value;
  // The checkmark icon's visibility is driven by CSS scoped to .selected
  // (style.css) — toggling this class alone is enough.
  document.querySelectorAll(".rotate-thumb").forEach((btn) => {
    btn.classList.toggle("selected", Number(btn.dataset.rotate) === value);
  });
}

document.querySelectorAll(".rotate-thumb").forEach((btn) => {
  btn.addEventListener("click", () => setSelectedRotation(Number(btn.dataset.rotate)));
});

function setRotateThumbPlaceholders(text) {
  document.querySelectorAll(".rotate-thumb-canvas-wrap").forEach((wrap) => {
    wrap.innerHTML = `<span class="rotate-thumb-placeholder">${text}</span>`;
  });
}

async function generateRotationPreviews(file) {
  setSelectedRotation(0);
  setRotateThumbPlaceholders("…");

  const objectUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = objectUrl;

  try {
    await new Promise((resolve, reject) => {
      video.onloadeddata = () => resolve();
      video.onerror = () => reject(new Error("video önizlemesi yüklenemedi"));
    });
    await new Promise((resolve) => {
      video.onseeked = () => resolve();
      // A tiny offset (not literally frame 0) avoids an undecoded/black
      // first frame on some codecs; clamp to half the duration for very
      // short clips.
      video.currentTime = Math.min(0.1, (video.duration || 0.2) / 2);
    });

    const maxDim = 220;
    const scale = Math.min(1, maxDim / Math.max(video.videoWidth, video.videoHeight));
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = Math.round(video.videoWidth * scale);
    sourceCanvas.height = Math.round(video.videoHeight * scale);
    sourceCanvas.getContext("2d").drawImage(video, 0, 0, sourceCanvas.width, sourceCanvas.height);

    for (const rotation of ROTATION_VALUES) {
      const wrap = document.querySelector(
        `.rotate-thumb[data-rotate="${rotation}"] .rotate-thumb-canvas-wrap`
      );
      const destCanvas = document.createElement("canvas");
      drawRotated(sourceCanvas, destCanvas, rotation);
      wrap.innerHTML = "";
      wrap.appendChild(destCanvas);
    }
  } catch (err) {
    setRotateThumbPlaceholders("önizleme yok");
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

setupDropzone("video-dropzone", "video-input", "video-submit", (file) => {
  selectedVideoFile = file;
  document.getElementById("video-dropzone").querySelector(".dropzone-text").textContent = file.name;
  generateRotationPreviews(file);
});

function updateSampleIntervalHint() {
  const select = document.getElementById("video-sample-interval");
  const hint = document.getElementById("sample-interval-hint");
  if (select.value) {
    hint.innerHTML =
      '<i data-lucide="triangle-alert"></i><span>Bu videoda gerçek plakalar sadece ~0.15 saniyelik dar bir ' +
      "pencerede okunabilir çıktı — saniye-bazlı örnekleme böyle kısa pencereleri tamamen " +
      "kaçırabilir. Sadece hızlı bir önizleme istiyorsanız kullanın.</span>";
    hint.classList.add("warning");
    refreshIcons();
  } else {
    hint.textContent = "";
    hint.classList.remove("warning");
  }
}
document.getElementById("video-sample-interval").addEventListener("change", updateSampleIntervalHint);
updateSampleIntervalHint();

document.getElementById("video-submit").addEventListener("click", async () => {
  if (!selectedVideoFile) return;
  const submitBtn = document.getElementById("video-submit");
  submitBtn.disabled = true;
  document.getElementById("video-result").classList.add("hidden");
  document.getElementById("video-progress-card").classList.remove("hidden");
  document.getElementById("video-status").textContent = "Video yükleniyor…";

  try {
    const formData = new FormData();
    formData.append("file", selectedVideoFile);
    formData.append("rotate", String(selectedRotation));
    const sampleInterval = document.getElementById("video-sample-interval").value;
    if (sampleInterval) formData.append("sample_interval_seconds", sampleInterval);
    const res = await fetch("/api/infer/video", { method: "POST", body: formData });
    if (!res.ok) throw new Error(`sunucu hatası (${res.status})`);
    const { job_id } = await res.json();
    pollVideoJob(job_id);
  } catch (err) {
    document.getElementById("video-status").textContent = `Hata: ${err.message}`;
    submitBtn.disabled = false;
  }
});

function formatEta(seconds) {
  if (seconds === null || seconds === undefined) return "";
  if (seconds < 1) return "birkaç saniye kaldı";
  if (seconds < 60) return `~${Math.round(seconds)} saniye kaldı`;
  return `~${Math.round(seconds / 60)} dakika kaldı`;
}

function pollVideoJob(jobId) {
  clearInterval(videoPollTimer);
  videoPollTimer = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) {
      clearInterval(videoPollTimer);
      document.getElementById("video-status").textContent = "Hata: iş bulunamadı.";
      return;
    }
    const job = await res.json();
    const pct = Math.round(job.progress * 100);
    document.getElementById("video-progress-fill").style.width = `${pct}%`;
    document.getElementById("video-progress-label").textContent = `%${pct}`;
    document.getElementById("video-status").textContent =
      `${job.frames_processed}/${job.total_frames || "?"} kare işlendi…`;
    document.getElementById("video-eta").textContent = formatEta(job.estimated_seconds_remaining);

    if (job.status === "done") {
      clearInterval(videoPollTimer);
      document.getElementById("video-submit").disabled = false;
      showVideoResult(job);
    } else if (job.status === "error") {
      clearInterval(videoPollTimer);
      document.getElementById("video-submit").disabled = false;
      document.getElementById("video-status").textContent = `Hata: ${job.error_message}`;
    }
  }, 1000);
}

function showVideoResult(job) {
  document.getElementById("video-progress-card").classList.add("hidden");
  document.getElementById("video-result").classList.remove("hidden");

  const summaryEl = document.getElementById("video-summary");
  summaryEl.innerHTML = "";
  // vehicle_type_counts is per distinct tracked vehicle, not per raw
  // frame detection (see docs/decisions.md #39) — so this sum is already
  // "how many vehicles passed", not "how many frames had a vehicle in them".
  const totalVehicles = Object.values(job.vehicle_type_counts).reduce((a, b) => a + b, 0);
  const readCount = job.vehicle_sightings.filter((s) => s.plate_status === "read").length;
  // Headline stats (red) vs per-type breakdown (blue) — a deliberate
  // two-tone split rather than an all-red summary row.
  const headlineStats = [["Tespit Edilen Araç Sayısı", totalVehicles], ["Benzersiz Plaka", readCount]];
  const breakdownStats = Object.entries(job.vehicle_type_counts).map(([type, count]) => [
    vehicleTypeLabel(type),
    count,
  ]);
  for (const [label, num] of headlineStats) {
    const div = document.createElement("div");
    div.className = "summary-stat";
    div.innerHTML = `<div class="num">${num}</div><div class="label">${label}</div>`;
    summaryEl.appendChild(div);
  }
  for (const [label, num] of breakdownStats) {
    const div = document.createElement("div");
    div.className = "summary-stat accent-blue";
    div.innerHTML = `<div class="num">${num}</div><div class="label">${label}</div>`;
    summaryEl.appendChild(div);
  }

  document.getElementById("video-download").href = job.video_url;

  const gallery = document.getElementById("video-plate-gallery");
  gallery.innerHTML = "";
  if (!job.vehicle_sightings.length) {
    gallery.innerHTML = '<p class="muted">Bu videoda hiç araç tespit edilmedi.</p>';
  }
  // Read first, then unreadable/no-plate — matches how job.vehicle_sightings
  // is already ordered server-side (docs/decisions.md #40), kept here too
  // in case a future caller reorders the array.
  const ordered = [...job.vehicle_sightings].sort((a, b) => {
    const rank = (s) => (s.plate_status === "read" ? 0 : 1);
    return rank(a) - rank(b);
  });
  for (const sighting of ordered) {
    const item = document.createElement("div");
    item.className = "plate-gallery-item";
    let plateRowHtml;
    if (sighting.plate_status === "read") {
      plateRowHtml = `<span class="plate-pill">${sighting.plate_text}</span>`;
    } else {
      const isUnreadable = sighting.plate_status === "unreadable";
      const label = isUnreadable ? "Plaka okunamadı" : "Plaka görünmüyor";
      const icon = isUnreadable ? "circle-help" : "eye-off";
      plateRowHtml = `<span class="status-pill"><i data-lucide="${icon}"></i>${label}</span>`;
      if (sighting.raw_ocr_text) {
        plateRowHtml += `<span class="plate-raw-ocr">ham okuma: "${sighting.raw_ocr_text}"</span>`;
      }
    }
    const typeInfo = VEHICLE_TYPE_INFO[sighting.vehicle_type] || { label: sighting.vehicle_type, icon: "help-circle" };
    item.innerHTML = `
      <img src="${sighting.thumbnail_url}" alt="${sighting.plate_text || typeInfo.label}" />
      <div class="plate-gallery-caption">
        <div class="plate-gallery-plate-row">${plateRowHtml}</div>
        <span class="plate-gallery-meta">
          <i data-lucide="${typeInfo.icon}"></i>${typeInfo.label}
          <i data-lucide="clock"></i>${sighting.timestamp_seconds.toFixed(1)}s
        </span>
      </div>`;
    gallery.appendChild(item);
  }
  refreshIcons();
}

// ---------- camera ----------
let cameraStream = null;
let cameraSocket = null;
let cameraRunning = false;
let lastAnnotatedUrl = null;

document.getElementById("camera-start").addEventListener("click", startCamera);
document.getElementById("camera-stop").addEventListener("click", stopCamera);

async function startCamera() {
  const statusEl = document.getElementById("camera-status");
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640 } });
  } catch (err) {
    statusEl.textContent = `Kameraya erişilemedi: ${err.message}`;
    return;
  }

  const rawVideo = document.getElementById("camera-raw");
  rawVideo.srcObject = cameraStream;
  await rawVideo.play();

  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  cameraSocket = new WebSocket(`${proto}://${window.location.host}/ws/camera`);
  cameraSocket.binaryType = "blob";

  cameraSocket.onopen = () => {
    cameraRunning = true;
    document.getElementById("camera-start").classList.add("hidden");
    document.getElementById("camera-stop").classList.remove("hidden");
    document.getElementById("camera-placeholder").classList.add("hidden");
    document.getElementById("camera-annotated").hidden = false;
    statusEl.textContent = "Bağlandı — canlı analiz çalışıyor.";
    sendNextCameraFrame();
  };

  cameraSocket.onmessage = (event) => {
    if (typeof event.data === "string") {
      const payload = JSON.parse(event.data);
      renderVehicleRows(document.querySelector("#camera-results-table tbody"), payload.vehicles);
      return;
    }
    const url = URL.createObjectURL(event.data);
    const img = document.getElementById("camera-annotated");
    img.src = url;
    if (lastAnnotatedUrl) URL.revokeObjectURL(lastAnnotatedUrl);
    lastAnnotatedUrl = url;
    if (cameraRunning) sendNextCameraFrame();
  };

  cameraSocket.onerror = () => {
    statusEl.textContent = "Bağlantı hatası.";
  };

  cameraSocket.onclose = () => {
    if (cameraRunning) statusEl.textContent = "Bağlantı kapandı.";
  };
}

function sendNextCameraFrame() {
  if (!cameraRunning || !cameraSocket || cameraSocket.readyState !== WebSocket.OPEN) return;
  const rawVideo = document.getElementById("camera-raw");
  const canvas = document.getElementById("camera-canvas");
  if (!rawVideo.videoWidth) {
    requestAnimationFrame(sendNextCameraFrame);
    return;
  }
  canvas.width = rawVideo.videoWidth;
  canvas.height = rawVideo.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(rawVideo, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(
    (blob) => {
      if (blob && cameraRunning && cameraSocket.readyState === WebSocket.OPEN) {
        cameraSocket.send(blob);
      }
    },
    "image/jpeg",
    0.75
  );
}

function stopCamera() {
  cameraRunning = false;
  if (cameraSocket) {
    cameraSocket.close();
    cameraSocket = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  document.getElementById("camera-start").classList.remove("hidden");
  document.getElementById("camera-stop").classList.add("hidden");
  document.getElementById("camera-annotated").hidden = true;
  document.getElementById("camera-placeholder").classList.remove("hidden");
  document.getElementById("camera-status").textContent = "";
  document.querySelector("#camera-results-table tbody").innerHTML = "";
}
