/* ═══════════════════════════════════════════════════════════════════════════════
   FlashGuard — app.js  (patched: element IDs aligned with dashboard.html)
   ═══════════════════════════════════════════════════════════════════════════════ */

const API = "";
let priceChart    = null;
let riskHistoryChart = null;
let portfolioChart   = null;
let autoRefreshActive = false;
let countdown        = 0;
let countdownTimer   = null;
let selectedFile     = null;
const riskHistory    = [];
const REFRESH_INTERVAL = 60;
let currentInterval  = "1m";

// ──────────────────────────────────────────────────────────────────────────────
// INIT
// ──────────────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initControls();
    initUploadZone();
    initNavHighlight();
    loadModels();
});

// ──────────────────────────────────────────────────────────────────────────────
// TAB NAVIGATION
// ──────────────────────────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll(".app-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            const page = tab.dataset.page;
            document.querySelectorAll(".app-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            document.querySelectorAll(".app-page").forEach(p => p.classList.remove("active"));
            document.getElementById(`page-${page}`)?.classList.add("active");
        });
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// NAVBAR SCROLL HIGHLIGHT
// ──────────────────────────────────────────────────────────────────────────────
function initNavHighlight() {
    const nb = document.getElementById("nb");
    if (nb) {
        window.addEventListener("scroll", () => {
            nb.classList.toggle("scrolled", window.scrollY > 28);
        });
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CONTROLS
// ──────────────────────────────────────────────────────────────────────────────
function initControls() {
    const presetSel = document.getElementById("preset-select");
    if (presetSel) {
        presetSel.addEventListener("change", e => {
            if (e.target.value) document.getElementById("ticker-input").value = e.target.value;
        });
    }

    const tickerInput = document.getElementById("ticker-input");
    if (tickerInput) {
        tickerInput.addEventListener("keydown", e => {
            if (e.key === "Enter") runPrediction();
        });
    }

    document.querySelectorAll(".pill-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".pill-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentInterval = btn.dataset.interval;
        });
    });

    document.getElementById("btn-auto")?.addEventListener("click", toggleAutoRefresh);
    document.getElementById("btn-predict")?.addEventListener("click", () => runPrediction());
    document.getElementById("btn-portfolio")?.addEventListener("click", () => runPortfolio());
    document.getElementById("btn-upload")?.addEventListener("click", () => runUpload());
}

function initUploadZone() {
    const zone  = document.getElementById("upload-zone");
    const input = document.getElementById("csv-file-input");
    if (!zone || !input) return;
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
    zone.addEventListener("dragleave", ()  => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", e => {
        e.preventDefault(); zone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) setSelectedFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => { if (input.files.length) setSelectedFile(input.files[0]); });
}

function setSelectedFile(file) {
    selectedFile = file;
    const zone = document.getElementById("upload-zone");
    zone.classList.add("has-file");
    document.getElementById("upload-content").innerHTML = `
        <span class="upload-icon">✅</span>
        <span class="upload-filename">${file.name}</span>
        <span class="upload-hint">${(file.size / 1024).toFixed(1)} KB · Click to change</span>`;
}

// ──────────────────────────────────────────────────────────────────────────────
// MODELS + CONNECTION
// ──────────────────────────────────────────────────────────────────────────────
async function loadModels() {
    try {
        const res    = await fetch(`${API}/api/models`);
        const models = await res.json();
        const sel    = document.getElementById("model-select");
        sel.innerHTML = "";
        [...models]
            .sort((a, b) => a.name.includes("minute") ? -1 : b.name.includes("minute") ? 1 : 0)
            .forEach(m => {
                const opt = document.createElement("option");
                opt.value       = m.name;
                opt.textContent = `${m.name.replace(".keras","").replace(".h5","")} (${m.timesteps}×${m.features})`;
                sel.appendChild(opt);
            });
        const sbModel = document.getElementById("sb-model");
        if (sbModel && sel.options.length > 0)
            sbModel.textContent = sel.options[0].text.split(" ")[0].toUpperCase();
        setConnection(true);
    } catch {
        showToast("Cannot connect to API — is it running on port 5000?");
        setConnection(false);
    }
}

function setConnection(on) {
    const dot = document.getElementById("conn-dot");
    const txt = document.getElementById("conn-text");
    if (dot) {
        dot.style.background  = on ? "var(--safe)" : "var(--danger)";
        dot.style.boxShadow   = on ? "0 0 8px rgba(0,255,157,0.5)" : "none";
        dot.classList.toggle("live", on);
    }
    if (txt) txt.textContent = on ? "API CONNECTED" : "API OFFLINE";

    const apiDot    = document.getElementById("api-dot");
    const apiStatus = document.getElementById("api-status");
    if (apiDot)    apiDot.className = "dot " + (on ? "on" : "off");
    if (apiStatus) apiStatus.textContent = on ? "API ONLINE" : "API OFFLINE";
}

// ──────────────────────────────────────────────────────────────────────────────
// LIVE PREDICTION
// ──────────────────────────────────────────────────────────────────────────────
async function runPrediction(silent = false) {
    const ticker = document.getElementById("ticker-input")?.value.trim();
    const model  = document.getElementById("model-select")?.value;
    if (!ticker) return showToast("Enter a ticker symbol");
    if (!model)  return showToast("No model loaded");

    const periodMap = { "1m":"5d","30m":"1mo","1d":"6mo","1wk":"2y","1mo":"5y" };
    if (!silent) showLoading("monitor");

    try {
        const upstox_token = document.getElementById("upstox-token")?.value.trim() || null;
        const res  = await fetch(`${API}/api/predict`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                ticker, model,
                period:       periodMap[currentInterval] || "6mo",
                interval:     currentInterval,
                threshold:    0.20,
                upstox_token: upstox_token || null,
            }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderLive(data);
        setConnection(true);
    } catch(err) {
        showToast("Prediction failed: " + err.message);
        if (!silent) hideLoading("monitor");
        setConnection(false);
    }
}

function renderLive(data) {
    hideLoading("monitor");
    hide("monitor-empty");
    show("monitor-results");

    const pct  = data.risk_pct;
    const band = data.band;

    // Risk gauge
    drawRiskRing(data.probability, "risk-ring");
    animateNumber(document.getElementById("risk-pct"), pct);
    document.getElementById("risk-pct").style.color = bandHex(band);

    setBadge("risk-badge", band);

    // Risk description — id in HTML is "risk-desc" not "risk-description"
    const descEl = document.getElementById("risk-desc");
    if (descEl) descEl.textContent =
        band === "STABLE"   ? "No significant crash risk detected" :
        band === "ELEVATED" ? "Elevated volatility — monitor closely" :
                              "⚠️ High crash probability detected!";

    // Stat cards — align with actual IDs in dashboard.html
    setText("stat-close",    "₹" + data.latest_close.toLocaleString("en-IN"));
    setText("stat-return",   (data.latest_close > 0 ? "—" : "—")); // no return field; placeholder
    setText("stat-date",     data.latest_date);
    setText("stat-model",    data.model.replace(".keras","").replace(".h5",""));
    setText("stat-interval", currentInterval.toUpperCase());

    const srcLabel = data.source === "upstox"   ? "Upstox (Live)"
                   : data.source === "yfinance"  ? "yFinance"
                   : "Demo Data";
    setText("stat-source", srcLabel);
    setText("chart-subtitle", `${data.ohlc.length} bars · ${currentInterval}`);
    setText("risk-timestamp", `AS OF ${data.latest_date}`);

    drawPriceChart(data.ohlc);

    riskHistory.push({
        time: new Date().toLocaleTimeString("en-IN", { hour:"2-digit", minute:"2-digit", second:"2-digit" }),
        risk: pct
    });
    if (riskHistory.length > 30) riskHistory.shift();
    drawRiskHistory();
}

// ──────────────────────────────────────────────────────────────────────────────
// PORTFOLIO
// ──────────────────────────────────────────────────────────────────────────────
async function runPortfolio() {
    const tickersRaw = document.getElementById("portfolio-tickers")?.value.trim();
    const model      = document.getElementById("model-select")?.value;
    const interval   = document.getElementById("portfolio-interval")?.value || "1d";
    if (!tickersRaw) return showToast("Enter tickers");
    const tickers    = tickersRaw.split(",").map(t => t.trim()).filter(Boolean);
    const periodMap  = { "1m":"5d","30m":"1mo","1d":"6mo","1wk":"2y","1mo":"5y" };
    showLoading("portfolio");
    try {
        const upstox_token = document.getElementById("upstox-token")?.value.trim() || null;
        const res  = await fetch(`${API}/api/portfolio`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                tickers, model,
                period:       periodMap[interval] || "6mo",
                interval,
                threshold:    0.20,
                upstox_token: upstox_token || null,
            }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderPortfolio(data);
    } catch(err) {
        showToast("Portfolio scan failed: " + err.message);
        hideLoading("portfolio");
    }
}

function renderPortfolio(data) {
    hideLoading("portfolio");
    hide("portfolio-empty");
    show("portfolio-results");
    setText("portfolio-subtitle", `${data.results.length} tickers · ${data.model.replace(".keras","").replace(".h5","")}`);

    let html = `<table class="data-table">
        <thead><tr>
            <th>Ticker</th><th>Price</th><th>Risk %</th>
            <th class="risk-bar-cell">Level</th><th>Status</th>
        </tr></thead><tbody>`;

    data.results.forEach(r => {
        if (r.error) {
            html += `<tr><td style="font-weight:600">${r.ticker.replace(".NS","")}</td>
                     <td colspan="4" style="color:var(--danger)">${r.error}</td></tr>`;
            return;
        }
        const bc = bandClass(r.band), color = bandHex(r.band);
        html += `<tr>
            <td style="font-weight:700">${r.ticker.replace(".NS","")}</td>
            <td class="mono">₹${r.latest_close.toLocaleString("en-IN")}</td>
            <td class="mono" style="font-weight:700;color:${color}">${r.risk_pct.toFixed(2)}%</td>
            <td class="risk-bar-cell">
                <div class="risk-bar-bg">
                    <div class="risk-bar-fill" style="width:${Math.min(r.risk_pct * 5, 100)}%;background:${color}"></div>
                </div>
            </td>
            <td><span class="band-pill ${bc}">${r.band}</span></td>
        </tr>`;
    });
    html += `</tbody></table>`;
    document.getElementById("portfolio-table-wrapper").innerHTML = html;

    const valid = data.results.filter(r => !r.error).sort((a,b) => b.probability - a.probability);
    const ctx   = document.getElementById("portfolio-chart")?.getContext("2d");
    if (!ctx) return;
    if (portfolioChart) portfolioChart.destroy();
    portfolioChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels:   valid.map(r => r.ticker.replace(".NS","")),
            datasets: [{
                data:              valid.map(r => r.risk_pct),
                backgroundColor:   valid.map(r => r.band==="STABLE" ? "rgba(0,217,126,0.45)" : r.band==="ELEVATED" ? "rgba(245,158,11,0.45)" : "rgba(255,61,46,0.45)"),
                borderColor:       valid.map(r => r.band==="STABLE" ? "#00d97e"              : r.band==="ELEVATED" ? "#f59e0b"               : "#ff3d2e"),
                borderWidth:       1.5,
                borderRadius:      4,
                hoverBackgroundColor: valid.map(r => r.band==="STABLE" ? "rgba(0,217,126,0.8)" : r.band==="ELEVATED" ? "rgba(245,158,11,0.8)" : "rgba(255,61,46,0.8)"),
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false, indexAxis: "y",
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(10,22,40,0.97)",
                    titleColor: "#e8f4ff", bodyColor: "#7ba8cc",
                    cornerRadius: 4, borderColor: "rgba(0,210,255,0.2)", borderWidth: 1, padding: 10,
                    titleFont: { family: "'Share Tech Mono', monospace" },
                    bodyFont:  { family: "'Share Tech Mono', monospace" },
                    callbacks: { label: c => `Risk: ${c.raw.toFixed(2)}%` },
                },
            },
            scales: {
                x: { min: 0, max: 100, grid: { color: "rgba(0,210,255,0.05)" }, ticks: { color: "#7ba8cc", font: { family: "'Share Tech Mono', monospace", size: 11 }, callback: v => v + "%" } },
                y: { grid: { display: false }, ticks: { color: "#e8f4ff", font: { family: "'Share Tech Mono', monospace", weight: "500", size: 11 } } },
            },
        },
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// CSV UPLOAD
// ──────────────────────────────────────────────────────────────────────────────
async function runUpload() {
    if (!selectedFile) return showToast("Select a CSV first");
    const model = document.getElementById("model-select")?.value;
    showLoading("upload");
    try {
        const fd = new FormData();
        fd.append("file", selectedFile);
        fd.append("model", model);
        fd.append("threshold", "0.20");
        const res  = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderUpload(data);
    } catch(err) {
        showToast("CSV failed: " + err.message);
        hideLoading("upload");
    }
}

function renderUpload(data) {
    hideLoading("upload");
    hide("upload-empty");
    show("upload-results");

    const pct  = data.risk_pct;
    const band = data.band;

    drawRiskRing(data.probability, "upload-risk-ring");
    animateNumber(document.getElementById("upload-risk-pct"), pct);
    document.getElementById("upload-risk-pct").style.color = bandHex(band);
    setBadge("upload-risk-badge", band);

    const descEl = document.getElementById("upload-risk-desc");
    if (descEl) descEl.textContent =
        band === "STABLE"   ? "No crash risk detected in CSV"   :
        band === "ELEVATED" ? "Elevated volatility in CSV data" :
                              "⚠️ High crash probability!";

    document.getElementById("upload-meta").innerHTML = `
        <div class="stat-card"><div class="stat-icon">📄</div><div class="stat-info"><span class="stat-label">File</span><span class="stat-value">${selectedFile.name}</span></div></div>
        <div class="stat-card"><div class="stat-icon">📊</div><div class="stat-info"><span class="stat-label">Rows</span><span class="stat-value">${data.rows_loaded.toLocaleString()}</span></div></div>
        <div class="stat-card"><div class="stat-icon">🧠</div><div class="stat-info"><span class="stat-label">Model</span><span class="stat-value mono">${data.model.replace(".keras","").replace(".h5","")}</span></div></div>
        <div class="stat-card"><div class="stat-icon">🎯</div><div class="stat-info"><span class="stat-label">Threshold</span><span class="stat-value">${(data.threshold*100).toFixed(0)}%</span></div></div>`;

    if (data.features && Object.keys(data.features).length) {
        let html = `<table class="feature-table">`;
        for (const [k, v] of Object.entries(data.features))
            html += `<tr><td>${k}</td><td>${typeof v === "number" ? v.toFixed(6) : v}</td></tr>`;
        html += `</table>`;
        document.getElementById("upload-features-wrapper").innerHTML = html;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DRAWING — risk ring
// ──────────────────────────────────────────────────────────────────────────────
function animateNumber(el, target) {
    if (!el) return;
    const start = parseFloat(el.textContent) || 0;
    const diff  = target - start;
    const dur   = 700;
    const t0    = performance.now();
    function tick(now) {
        const p = Math.min((now - t0) / dur, 1);
        const e = 1 - Math.pow(1 - p, 3);
        el.textContent = (start + diff * e).toFixed(1);
        if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function drawRiskRing(prob, id) {
    const canvas = document.getElementById(id);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const sz  = 200;
    canvas.width  = sz * dpr; canvas.height = sz * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width  = sz + "px";
    canvas.style.height = sz + "px";

    const cx = sz / 2, cy = sz / 2, r = 82, lw = 8;
    const sa = -0.5 * Math.PI, fa = 2 * Math.PI;

    // Track
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, fa);
    ctx.lineWidth = lw; ctx.strokeStyle = "rgba(255,255,255,0.04)"; ctx.stroke();

    // Tick marks
    for (let i = 0; i < 60; i++) {
        const a = sa + (i / 60) * fa, l = i % 5 === 0 ? 7 : 3;
        const x1 = cx + (r+5)*Math.cos(a), y1 = cy + (r+5)*Math.sin(a);
        const x2 = cx + (r+5+l)*Math.cos(a), y2 = cy + (r+5+l)*Math.sin(a);
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.lineWidth = i % 5 === 0 ? 1.5 : 0.5;
        ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.stroke();
    }

    // Arc
    const ea   = sa + prob * fa;
    const grad = ctx.createConicGradient(sa, cx, cy);
    grad.addColorStop(0,   "#22c55e");
    grad.addColorStop(0.3, "#eab308");
    grad.addColorStop(0.6, "#ef4444");
    grad.addColorStop(1,   "#ef4444");
    ctx.beginPath(); ctx.arc(cx, cy, r, sa, ea);
    ctx.lineWidth = lw; ctx.strokeStyle = grad; ctx.lineCap = "round"; ctx.stroke();

    // Glow dot
    if (prob > 0.01) {
        const ex = cx + r * Math.cos(ea), ey = cy + r * Math.sin(ea);
        const g  = ctx.createRadialGradient(ex, ey, 0, ex, ey, 12);
        const c  = prob < 0.13 ? "34,197,94" : prob < 0.5 ? "234,179,8" : "239,68,68";
        g.addColorStop(0, `rgba(${c},0.5)`);
        g.addColorStop(1, `rgba(${c},0)`);
        ctx.beginPath(); ctx.arc(ex, ey, 12, 0, fa);
        ctx.fillStyle = g; ctx.fill();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DRAWING — TradingView Lightweight Charts candlestick
// ──────────────────────────────────────────────────────────────────────────────
let tvChart      = null;
let candleSeries = null;
let volumeSeries = null;
let smaSeries    = null;

function drawPriceChart(ohlc) {
    const container = document.getElementById("price-chart");
    if (!container) return;

    if (!tvChart) {
        tvChart = LightweightCharts.createChart(container, {
            layout: {
                background: { type: "solid", color: "transparent" },
                textColor:  "#7ba8cc",
                fontFamily: "'Share Tech Mono', monospace",
            },
            grid: {
                vertLines: { color: "rgba(0,210,255,0.05)" },
                horzLines: { color: "rgba(0,210,255,0.05)" },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: "rgba(0,210,255,0.3)", width: 1, style: 3 },
                horzLine: { color: "rgba(0,210,255,0.3)", width: 1, style: 3 },
            },
            rightPriceScale: { borderColor: "rgba(0,210,255,0.1)" },
            timeScale: {
                borderColor:    "rgba(0,210,255,0.1)",
                timeVisible:    true,
                secondsVisible: false,
                rightOffset:    12,
            },
            handleScroll: { mouseWheel: true, pressedMouseMove: true },
            handleScale:  { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
        });

        candleSeries = tvChart.addCandlestickSeries({
            upColor:        "#00ff9d", downColor:       "#ff3d2e",
            borderUpColor:  "#00ff9d", borderDownColor: "#ff3d2e",
            wickUpColor:    "#00ff9d", wickDownColor:   "#ff3d2e",
        });

        smaSeries = tvChart.addLineSeries({
            color: "rgba(0,210,255,0.85)", lineWidth: 2,
            crosshairMarkerVisible: false, priceLineVisible: false,
        });

        volumeSeries = tvChart.addHistogramSeries({
            color: "#26a69a",
            priceFormat: { type: "volume" },
            priceScaleId: "",
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        new ResizeObserver(entries => {
            if (!entries.length || entries[0].target !== container) return;
            const rect = entries[0].contentRect;
            tvChart.applyOptions({ height: rect.height, width: rect.width });
        }).observe(container);
    }

    ohlc.sort((a, b) => new Date(a.date) - new Date(b.date));

    const candleData = [];
    const volData    = [];

    ohlc.forEach(d => {
        let t = Math.floor(new Date(d.date).getTime() / 1000);
        if (candleData.length > 0 && t <= candleData[candleData.length - 1].time)
            t = candleData[candleData.length - 1].time + 60;

        candleData.push({ time: t, open: d.open, high: d.high, low: d.low, close: d.close });
        volData.push({ time: t, value: d.volume || 0, color: d.close >= d.open ? "rgba(0,217,126,0.4)" : "rgba(255,61,46,0.4)" });
    });

    const smaData = [];
    for (let i = 19; i < candleData.length; i++) {
        let sum = 0;
        for (let j = 0; j < 20; j++) sum += candleData[i - j].close;
        smaData.push({ time: candleData[i].time, value: sum / 20 });
    }

    candleSeries.setData(candleData);
    smaSeries.setData(smaData);
    volumeSeries.setData(volData);
    tvChart.timeScale().fitContent();
}

function drawRiskHistory() {
    const canvas = document.getElementById("risk-history-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (riskHistoryChart) riskHistoryChart.destroy();
    setText("risk-history-count", `${riskHistory.length} scan${riskHistory.length !== 1 ? "s" : ""}`);

    const grad = ctx.createLinearGradient(0, 0, 0, 120);
    grad.addColorStop(0, "rgba(0,210,255,0.3)");
    grad.addColorStop(1, "rgba(0,210,255,0)");

    riskHistoryChart = new Chart(ctx, {
        type: "line",
        data: {
            labels:   riskHistory.map(r => r.time),
            datasets: [{
                data:                 riskHistory.map(r => r.risk),
                borderColor:          "#00d2ff",
                backgroundColor:      grad,
                borderWidth:          2,
                pointRadius:          4,
                pointHoverRadius:     6,
                pointBackgroundColor: "#0a1628",
                pointBorderColor:     "#00d2ff",
                fill: true, tension: 0.4,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(10,22,40,0.97)",
                    titleColor: "#e8f4ff", bodyColor: "#7ba8cc",
                    cornerRadius: 4, borderColor: "rgba(0,210,255,0.2)", borderWidth: 1, padding: 10,
                    titleFont: { family: "'Share Tech Mono', monospace" },
                    bodyFont:  { family: "'Share Tech Mono', monospace" },
                    callbacks: { label: c => `RISK: ${c.raw.toFixed(2)}%` },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: "#3d6080", font: { family: "'Share Tech Mono', monospace", size: 9 }, maxTicksLimit: 5 } },
                y: { min: 0, max: 100, grid: { color: "rgba(0,210,255,0.05)" }, ticks: { color: "#7ba8cc", font: { family: "'Share Tech Mono', monospace" } } },
            },
        },
    });
}

// ──────────────────────────────────────────────────────────────────────────────
// AUTO-REFRESH
// ──────────────────────────────────────────────────────────────────────────────
function toggleAutoRefresh() {
    autoRefreshActive = !autoRefreshActive;
    const btn   = document.getElementById("btn-auto");
    const timer = document.getElementById("refresh-timer");
    if (autoRefreshActive) {
        btn?.classList.add("on");
        countdown = REFRESH_INTERVAL;
        clearInterval(countdownTimer);
        countdownTimer = setInterval(() => {
            countdown--;
            if (timer) timer.textContent = countdown > 0 ? `${countdown}s` : "…";
            if (countdown <= 0) { runPrediction(true); countdown = REFRESH_INTERVAL; }
        }, 1000);
        runPrediction();
    } else {
        btn?.classList.remove("on");
        clearInterval(countdownTimer);
        if (timer) timer.textContent = "";
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ──────────────────────────────────────────────────────────────────────────────
function bandHex(b)   { return b === "STABLE" ? "#22c55e" : b === "ELEVATED" ? "#eab308" : "#ef4444"; }
function bandClass(b) { return b === "STABLE" ? "stable"  : b === "ELEVATED" ? "elevated" : "high"; }

function setBadge(id, band) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = band;
    el.className   = `risk-badge ${bandClass(band)}`;
}

function setText(id, txt) {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
}
function show(id) { document.getElementById(id)?.classList.remove("hidden"); }
function hide(id) { document.getElementById(id)?.classList.add("hidden"); }

function showLoading(s) {
    hide(`${s}-results`);
    hide(`${s}-empty`);
    show(`${s}-loading`);
}
function hideLoading(s) { hide(`${s}-loading`); }

function showToast(msg) {
    const t = document.createElement("div");
    t.className   = "toast";
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => {
        t.style.opacity    = "0";
        t.style.transition = "opacity 0.3s";
        setTimeout(() => t.remove(), 300);
    }, 4000);
}
