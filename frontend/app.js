
const API = "http://localhost:8000";
const dropZone   = document.getElementById("drop-zone");
const fileInput  = document.getElementById("file-input");
const fileNameEl = document.getElementById("file-name");
const fsInput    = document.getElementById("fs-input");
const analyseBtn = document.getElementById("analyse-btn");
const loading    = document.getElementById("loading");
const resultCard = document.getElementById("result-card");
const resetBtn   = document.getElementById("reset-btn");
const dlBtn      = document.getElementById("download-result-csv-btn");

let selectedFile = null;
let lastGenClass = "Normal";

// Tabs
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + tab));
    resultCard.classList.add("hidden");
  });
});

// File input
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
analyseBtn.addEventListener("click", runAnalysis);
resetBtn.addEventListener("click", resetUI);
dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => { e.preventDefault(); dropZone.classList.remove("drag-over"); handleFile(e.dataTransfer.files[0]); });
dropZone.addEventListener("click", () => fileInput.click());

function handleFile(file) {
  if (!file) return;
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["csv","mat"].includes(ext)) { alert("Please select a .csv or .mat file."); return; }
  selectedFile = file;
  fileNameEl.textContent = "Selected: " + file.name;
  analyseBtn.disabled = false;
}

// Demo buttons
document.querySelectorAll(".btn-demo").forEach(btn => {
  btn.addEventListener("click", async () => {
    lastGenClass = btn.dataset.class;
    showLoading();
    try {
      const res = await fetch(API + "/generate-demo/" + encodeURIComponent(lastGenClass));
      if (!res.ok) throw new Error(await res.text());
      renderResult(await res.json(), false);
    } catch(e) { hideLoading(); alert("Error: " + e.message); }
  });
});

async function runAnalysis() {
  if (!selectedFile) return;
  showLoading();
  const form = new FormData();
  form.append("file", selectedFile);
  const fs = parseFloat(fsInput.value);
  if (fs > 0) form.append("fs", fs);
  try {
    const res = await fetch(API + "/diagnose", { method:"POST", body:form });
    if (!res.ok) { const e = await res.json().catch(()=>({detail:res.statusText})); throw new Error(e.detail||res.statusText); }
    renderResult(await res.json(), false);
  } catch(e) { hideLoading(); alert("Analysis failed: " + e.message); }
}

// Generator tab
let selectedGenClass = "Normal";
document.querySelectorAll(".gen-class-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".gen-class-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    selectedGenClass = btn.dataset.class;
  });
});

document.getElementById("gen-classify-btn").addEventListener("click", async () => {
  lastGenClass = selectedGenClass;
  showLoading();
  try {
    const res = await fetch(API + "/generate-demo/" + encodeURIComponent(selectedGenClass));
    if (!res.ok) throw new Error(await res.text());
    renderResult(await res.json(), true);
  } catch(e) { hideLoading(); alert("Error: " + e.message); }
});

document.getElementById("gen-download-btn").addEventListener("click", () => {
  window.location.href = API + "/generate-signal/" + encodeURIComponent(selectedGenClass);
});

dlBtn.addEventListener("click", () => {
  window.location.href = API + "/generate-signal/" + encodeURIComponent(lastGenClass);
});

// Render
function renderResult(data, fromGenerator) {
  hideLoading();
  const cls    = data.predicted_class;
  const clsKey = cls.replace(/\s+/g, "-");
  const icons  = {"Normal":"&#9989;","High Vibration":"&#9888;","Harmonic Fault":"&#128260;","Impulsive Fault":"&#128165;"};
  document.getElementById("verdict-banner").className = "verdict-banner verdict-" + clsKey;
  document.getElementById("verdict-icon").innerHTML   = icons[cls] || "&#128269;";
  document.getElementById("verdict-class").textContent = cls;
  const urgency = (data.ai && data.ai.urgency) ? data.ai.urgency : "Unknown";
  const uEl = document.getElementById("urgency-badge");
  uEl.textContent = urgency; uEl.className = "urgency-badge urgency-" + urgency;
  const si = data.signal_info || {};
  document.getElementById("signal-info").innerHTML = [
    si.n_samples  ? "<span><strong>" + si.n_samples  + "</strong> samples</span>" : "",
    si.fs_hz      ? "<span><strong>" + si.fs_hz.toFixed(0) + "</strong> Hz</span>" : "",
    si.duration_s ? "<span><strong>" + si.duration_s.toFixed(2) + "</strong> s</span>" : "",
    si.true_class ? "<span>Class: <strong>" + si.true_class + "</strong></span>" : "",
  ].join("");
  const wb = document.getElementById("warnings-box");
  if (data.warnings && data.warnings.length) {
    wb.innerHTML = "<ul>" + data.warnings.map(w => "<li>" + w + "</li>").join("") + "</ul>";
    wb.classList.remove("hidden");
  } else { wb.classList.add("hidden"); }
  const fl = {rms:"RMS",peak:"Peak",crest_factor:"Crest Factor",fundamental_freq_hz:"Fundamental Frequency (Hz)",second_harmonic_ratio:"2nd Harmonic Ratio"};
  document.getElementById("feature-rows").innerHTML = Object.entries(fl)
    .map(([k,l]) => "<tr><td>" + l + "</td><td>" + (data.features[k] !== undefined ? data.features[k] : "--") + "</td></tr>").join("");
  const scores = data.scores || {};
  const maxS = Math.max(...Object.values(scores));
  document.getElementById("score-bars").innerHTML = Object.entries(scores)
    .sort((a,b) => b[1]-a[1]).map(([name,val]) => {
      const pct = (val*100).toFixed(0);
      const cls2 = val === maxS ? " winner" : "";
      return "<div class='score-bar-row'><span class='score-bar-label'>" + name + "</span>" +
        "<div class='score-bar-track'><div class='score-bar-fill" + cls2 + "' style='width:" + pct + "%'></div></div>" +
        "<span class='score-bar-value'>" + val.toFixed(2) + "</span></div>";
    }).join("");
  document.getElementById("observations-list").innerHTML =
    ((data.interpretation && data.interpretation.observations) || []).map(o => "<li>" + o + "</li>").join("");
  const ai = data.ai || {};
  document.getElementById("ai-explanation").textContent =
    ai.explanation || (data.interpretation && data.interpretation.summary) || "";
  document.getElementById("ai-actions").innerHTML = (ai.actions||[]).map(a => "<li>" + a + "</li>").join("");
  document.getElementById("ai-unavailable").classList.toggle("hidden", !!ai.available);
  dlBtn.classList.toggle("hidden", !fromGenerator);
  if (fromGenerator && si.true_class) lastGenClass = si.true_class;
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({behavior:"smooth"});
}

function showLoading() { loading.classList.remove("hidden"); resultCard.classList.add("hidden"); }
function hideLoading()  { loading.classList.add("hidden"); }
function resetUI() {
  selectedFile = null; fileNameEl.textContent = "";
  fileInput.value = ""; analyseBtn.disabled = true;
  resultCard.classList.add("hidden"); loading.classList.add("hidden");
}
