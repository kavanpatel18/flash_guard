/* FlashGuard v4 — app.js
   TradingView-style UI. Upstox-first real-time data. Chart.js candlestick. */

const API = "";
const UPSTOX_TOKEN_KEY = "fg_upstox_token";

let currentInterval = "1d";
let currentTicker   = "RELIANCE";
let priceChart      = null;
let historyChart    = null;
let autoInterval    = null;
let autoActive      = false;
let selectedFile    = null;
const scanHistory   = [];

// ── Watchlist tickers ────────────────────────────────────────────────────────
const WATCHLIST = [
    {sym:"NIFTY50",  label:"NIFTY 50"},
    {sym:"BANKNIFTY",label:"BANK NIFTY"},
    {sym:"RELIANCE", label:"RELIANCE"},
    {sym:"TCS",      label:"TCS"},
    {sym:"HDFCBANK", label:"HDFC BANK"},
    {sym:"INFY",     label:"INFOSYS"},
    {sym:"ICICIBANK",label:"ICICI BANK"},
    {sym:"SBIN",     label:"SBI"},
    {sym:"WIPRO",    label:"WIPRO"},
    {sym:"MARUTI",   label:"MARUTI"},
];

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initTabs();
    initIntervalPills();
    initWatchlist();
    initTokenInput();
    initAutoRefresh();
    initCSV();
    loadModels();
    document.getElementById("btnScan").addEventListener("click", () => runScan());
    document.getElementById("tickerInput").addEventListener("keydown", e => {
        if (e.key === "Enter") runScan();
    });
    document.getElementById("btnPortfolio").addEventListener("click", runPortfolio);

    // Refresh watchlist prices every 15s
    refreshWatchlistPrices();
    setInterval(refreshWatchlistPrices, 15000);
});

// ── Clock ────────────────────────────────────────────────────────────────────
function initClock() {
    const el = document.getElementById("clock");
    const tick = () => {
        const n = new Date(), p = v => String(v).padStart(2,"0");
        el.textContent = `${p(n.getHours())}:${p(n.getMinutes())}:${p(n.getSeconds())}`;
    };
    tick(); setInterval(tick, 1000);
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll(".tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-body").forEach(b => {
                b.classList.remove("active");
                b.style.display = "none";
            });
            tab.classList.add("active");
            const body = document.querySelector(`.tab-body[data-tab="${tab.dataset.tab}"]`);
            if (body) {
                body.classList.add("active");
                body.style.display = "flex";
                body.style.flexDirection = "column";
                body.style.flex = "1";
                body.style.minHeight = "0";
            }
        });
    });
    // Ensure first tab is shown
    const first = document.querySelector(".tab-body[data-tab='chart']");
    if (first) { first.style.display = "flex"; first.style.flexDirection = "column"; first.style.flex = "1"; first.style.minHeight = "0"; }
}

// ── Interval pills ───────────────────────────────────────────────────────────
function initIntervalPills() {
    document.querySelectorAll(".pill").forEach(p => {
        p.addEventListener("click", () => {
            document.querySelectorAll(".pill").forEach(b => b.classList.remove("active"));
            p.classList.add("active");
            currentInterval = p.dataset.iv;
            setText("bInterval", currentInterval.toUpperCase());
        });
    });
}

// ── Token input ───────────────────────────────────────────────────────────────
function initTokenInput() {
    const inp = document.getElementById("tokenInput");
    // Restore from localStorage
    const saved = localStorage.getItem(UPSTOX_TOKEN_KEY);
    if (saved) { inp.value = saved; updateTokenStatus(saved); }

    inp.addEventListener("input", () => {
        const val = inp.value.trim();
        if (val) localStorage.setItem(UPSTOX_TOKEN_KEY, val);
        updateTokenStatus(val);
    });
}

function updateTokenStatus(token) {
    const el = document.getElementById("tokenStatus");
    if (!el) return;
    if (token && token.length > 20) {
        el.textContent = "✓ Token set — using Upstox live data";
        el.className   = "token-status ok";
    } else {
        el.textContent = "Enter token for live data";
        el.className   = "token-status";
    }
}

function getToken() {
    return document.getElementById("tokenInput")?.value.trim() || localStorage.getItem(UPSTOX_TOKEN_KEY) || "";
}

// ── Watchlist ─────────────────────────────────────────────────────────────────
function initWatchlist() {
    const el = document.getElementById("watchlist");
    el.innerHTML = WATCHLIST.map(w => `
        <div class="watch-item" data-sym="${w.sym}" onclick="selectWatchItem('${w.sym}')">
            <span class="watch-name">${w.label}</span>
            <span class="watch-price mono" id="wp-${w.sym}">—</span>
            <span class="watch-chg" id="wc-${w.sym}">—</span>
        </div>`).join("");
    // Set first item active
    document.querySelector(`.watch-item[data-sym="${currentTicker}"]`)?.classList.add("active");
}

function selectWatchItem(sym) {
    document.querySelectorAll(".watch-item").forEach(i => i.classList.remove("active"));
    document.querySelector(`.watch-item[data-sym="${sym}"]`)?.classList.add("active");
    currentTicker = sym;
    document.getElementById("tickerInput").value = sym;
    runScan();
}

async function refreshWatchlistPrices() {
    const token = getToken();
    if (!token) return;
    try {
        const res  = await fetch(`${API}/api/market-overview`, {
            method: "POST", headers: {"Content-Type":"application/json"},
            body: JSON.stringify({token})
        });
        const data = await res.json();
        if (data.data && data.data.length) {
            data.data.forEach(q => {
                // Match by name
                WATCHLIST.forEach(w => {
                    if (q.name.includes(w.label.split(" ")[0]) || q.name === w.label) {
                        const pEl = document.getElementById(`wp-${w.sym}`);
                        const cEl = document.getElementById(`wc-${w.sym}`);
                        if (pEl) pEl.textContent = "₹" + (q.ltp||0).toLocaleString("en-IN");
                        if (cEl) {
                            const up = (q.change_pct||0) >= 0;
                            cEl.textContent = (up?"+":"") + (q.change_pct||0).toFixed(2) + "%";
                            cEl.className   = "watch-chg " + (up?"up":"dn");
                        }
                    }
                });
            });
        }
    } catch(e) { /* watchlist refresh failed silently */ }
}

// ── Models ────────────────────────────────────────────────────────────────────
async function loadModels() {
    try {
        const models = await (await fetch(`${API}/api/models`)).json();
        const sel    = document.getElementById("modelSelect");
        sel.innerHTML = "";
        const valid  = models.filter(m => !m.error);
        if (!valid.length) {
            sel.innerHTML = "<option>No models loaded</option>";
            setConn(false); return;
        }
        valid.sort((a,b) => a.name.includes("minute")?-1:1).forEach(m => {
            const o = document.createElement("option");
            o.value = m.name;
            o.textContent = m.name.replace(".keras","").replace(".h5","") +
                            ` (${m.timesteps}×${m.features})`;
            sel.appendChild(o);
        });
        setConn(true);
    } catch(e) {
        showToast("Cannot connect to API — run api_server.py first");
        setConn(false);
    }
}

function setConn(on) {
    const dot = document.getElementById("connDot");
    const lbl = document.getElementById("connLabel");
    if (dot) dot.className = "conn-dot" + (on?" live":"");
    if (lbl) lbl.textContent = on ? "LIVE" : "OFFLINE";
}

// ── Auto refresh ──────────────────────────────────────────────────────────────
function initAutoRefresh() {
    const btn = document.getElementById("btnAuto");
    btn.addEventListener("click", () => {
        autoActive = !autoActive;
        if (autoActive) {
            btn.style.background = "#3fb950";
            btn.style.color      = "#000";
            btn.textContent      = "AUTO ✓";
            autoInterval = setInterval(() => runScan(true), 30000);
        } else {
            btn.style.background = "var(--bg4)";
            btn.style.color      = "var(--text2)";
            btn.textContent      = "AUTO";
            clearInterval(autoInterval);
        }
    });
}

// ── MAIN SCAN ────────────────────────────────────────────────────────────────
async function runScan(silent = false) {
    const ticker = document.getElementById("tickerInput")?.value.trim() || currentTicker;
    const model  = document.getElementById("modelSelect")?.value;
    const token  = getToken();
    if (!ticker) return showToast("Enter a ticker");
    if (!model)  return showToast("No model loaded");

    currentTicker = ticker;
    const periodMap = {"1m":"5d","30m":"1mo","1d":"6mo","1wk":"2y","1mo":"5y"};

    if (!silent) {
        // Show loading state
        document.getElementById("chartEmpty").classList.add("hidden");
        document.getElementById("priceChart").classList.add("hidden");
        document.getElementById("chartTicker").textContent = ticker;
        document.getElementById("chartPrice").textContent  = "Loading…";
    }

    try {
        const res  = await fetch(`${API}/api/predict`, {
            method: "POST", headers: {"Content-Type":"application/json"},
            body: JSON.stringify({
                ticker, model, token,
                period:    periodMap[currentInterval] || "6mo",
                interval:  currentInterval,
                threshold: 0.20,
            })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        renderScan(data);
        setConn(true);
    } catch(e) {
        showToast("Scan failed: " + e.message);
        document.getElementById("chartEmpty").classList.remove("hidden");
        setConn(false);
    }
}

function renderScan(data) {
    const pct  = data.risk_pct  ?? 0;
    const prob = data.probability ?? 0;
    const band = data.band ?? "STABLE";
    const ohlc = data.ohlc ?? [];

    // ── Chart header ──
    const last  = ohlc.length ? ohlc[ohlc.length-1] : null;
    const prev  = ohlc.length > 1 ? ohlc[ohlc.length-2] : null;
    const price = data.live_price || data.latest_close || (last?.close ?? 0);
    const chg   = last && prev ? last.close - prev.close : 0;
    const chgP  = prev && prev.close ? (chg/prev.close*100) : 0;
    const up    = chg >= 0;

    setText("chartTicker", data.ticker?.replace(".NS","") || currentTicker);
    setText("chartPrice",  "₹" + price.toLocaleString("en-IN", {minimumFractionDigits:2,maximumFractionDigits:2}));

    const chgEl = document.getElementById("chartChange");
    chgEl.textContent = `${up?"+":""}${chg.toFixed(2)} (${up?"+":""}${chgP.toFixed(2)}%)`;
    chgEl.className   = "chart-change " + (up?"up":"dn");

    if (last) {
        setText("ohlcO", "₹"+last.open.toLocaleString("en-IN"));
        setText("ohlcH", "₹"+last.high.toLocaleString("en-IN"));
        setText("ohlcL", "₹"+last.low.toLocaleString("en-IN"));
        setText("ohlcC", "₹"+last.close.toLocaleString("en-IN"));
        setText("ohlcV", (last.volume||0).toLocaleString());
    }
    setText("chartMeta", `${ohlc.length} bars · ${currentInterval.toUpperCase()} · ${data.latest_date??""}`);

    // ── Bottom strip ──
    setText("bClose",    "₹" + price.toLocaleString("en-IN"));
    setText("bInterval", currentInterval.toUpperCase());
    setText("bBars",     ohlc.length.toString());
    const srcMap = {upstox:"Upstox ●", yfinance:"yFinance", demo:"Demo"};
    setText("bSource",  srcMap[data.source] || data.source);
    setText("bUpdated", new Date().toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit",second:"2-digit"}));

    // ── Candlestick chart ──
    drawCandleChart(ohlc);

    // ── Risk gauge ──
    drawRiskRing(prob);
    animateNum(document.getElementById("gaugePct"), pct);

    const bandEl = document.getElementById("riskBand");
    bandEl.textContent = band;
    bandEl.className   = "risk-band " + bandClass(band);

    setText("riskDesc",
        band === "HIGH RISK" ? "⚠️ High crash probability detected!" :
        band === "ELEVATED"  ? "Elevated volatility — monitor closely" :
                               "No significant crash risk detected");

    // ── Info panel ──
    setText("infoTicker", data.ticker?.replace(".NS","") || currentTicker);
    setText("infoModel",  (data.model||"").replace(".keras","").replace(".h5",""));
    setText("infoTs",     data.timesteps?.toString() || "—");
    setText("infoFeat",   data.features?.toString()  || "—");
    setText("infoDate",   data.latest_date || "—");

    const srcEl  = document.getElementById("infoSource");
    const srcCls = {upstox:"upstox",yfinance:"yfinance",demo:"demo"}[data.source]||"demo";
    srcEl.innerHTML = `<span class="source-pill ${srcCls}">${srcMap[data.source]||data.source}</span>`;

    // ── History ──
    scanHistory.push({
        t: new Date().toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit",second:"2-digit"}),
        v: pct
    });
    if (scanHistory.length > 30) scanHistory.shift();
    drawHistory();

    // ── Watchlist active item price ──
    const wPEl = document.getElementById(`wp-${currentTicker}`);
    if (wPEl) wPEl.textContent = "₹" + price.toLocaleString("en-IN");
}

// ── Candlestick Chart ─────────────────────────────────────────────────────────
function drawCandleChart(ohlc) {
    document.getElementById("chartEmpty").classList.add("hidden");
    const canvas = document.getElementById("priceChart");
    canvas.classList.remove("hidden");

    if (priceChart) { priceChart.destroy(); priceChart = null; }
    if (!ohlc || !ohlc.length) return;

    const ctx = canvas.getContext("2d");

    // Colours
    const GREEN  = "#3fb950";
    const RED    = "#f85149";
    const GREENA = "rgba(63,185,80,0.7)";
    const REDA   = "rgba(248,81,73,0.7)";
    const GRID   = "rgba(48,54,61,0.6)";
    const TEXT   = "#8b949e";
    const MONO   = "'JetBrains Mono', monospace";

    // Build candlestick data as custom bars
    // Chart.js doesn't have a native candlestick type, so we use a Float bar plugin trick:
    // We draw wicks via a custom plugin and use bar for candle bodies
    const labels  = ohlc.map(d => d.date.slice(0,10));
    const opens   = ohlc.map(d => d.open);
    const highs   = ohlc.map(d => d.high);
    const lows    = ohlc.map(d => d.low);
    const closes  = ohlc.map(d => d.close);
    const volumes = ohlc.map(d => d.volume || 0);
    const colors  = ohlc.map(d => d.close >= d.open ? GREENA : REDA);
    const borders = ohlc.map(d => d.close >= d.open ? GREEN  : RED);

    // Candlestick body = [open, close] as a floating bar
    const candleData = ohlc.map(d => [
        Math.min(d.open, d.close),
        Math.max(d.open, d.close)
    ]);

    // Volume (separate y-axis)
    const maxVol = Math.max(...volumes);

    // Custom plugin to draw wicks
    const wickPlugin = {
        id: "wickPlugin",
        afterDatasetsDraw(chart) {
            const { ctx, data, scales } = chart;
            const ds  = data.datasets[0];
            const meta= chart.getDatasetMeta(0);
            if (!meta.visible) return;
            ctx.save();
            meta.data.forEach((bar, i) => {
                const x = bar.x;
                const yScale = scales["y"];
                const high  = yScale.getPixelForValue(highs[i]);
                const low   = yScale.getPixelForValue(lows[i]);
                const bodyT = Math.min(bar.y, bar.base);
                const bodyB = Math.max(bar.y, bar.base);
                ctx.beginPath();
                ctx.strokeStyle = borders[i];
                ctx.lineWidth   = 1;
                // Upper wick
                ctx.moveTo(x, high);
                ctx.lineTo(x, bodyT);
                // Lower wick
                ctx.moveTo(x, bodyB);
                ctx.lineTo(x, low);
                ctx.stroke();
            });
            ctx.restore();
        }
    };

    priceChart = new Chart(ctx, {
        type: "bar",
        plugins: [wickPlugin],
        data: {
            labels,
            datasets: [
                {
                    label:           "OHLC",
                    data:            candleData,
                    backgroundColor: colors,
                    borderColor:     borders,
                    borderWidth:     1,
                    borderSkipped:   false,
                    yAxisID:         "y",
                    order:           1,
                    barPercentage:   0.6,
                    categoryPercentage: 0.8,
                },
                {
                    type:            "bar",
                    label:           "Volume",
                    data:            volumes,
                    backgroundColor: ohlc.map(d => d.close>=d.open
                        ? "rgba(63,185,80,0.25)" : "rgba(248,81,73,0.25)"),
                    yAxisID:         "yVol",
                    order:           2,
                    barPercentage:   0.8,
                    categoryPercentage: 0.9,
                }
            ]
        },
        options: {
            responsive:          true,
            maintainAspectRatio: false,
            animation:           { duration: 300 },
            interaction:         { intersect: false, mode: "index" },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#161b22",
                    borderColor:     "#30363d",
                    borderWidth:     1,
                    titleColor:      "#e6edf3",
                    bodyColor:       "#8b949e",
                    cornerRadius:    6,
                    padding:         10,
                    titleFont:       { family: MONO, size: 11 },
                    bodyFont:        { family: MONO, size: 11 },
                    callbacks: {
                        title: items => labels[items[0].dataIndex],
                        label: item => {
                            if (item.dataset.label === "OHLC") {
                                const i = item.dataIndex;
                                return [
                                    `O: ₹${opens[i].toLocaleString("en-IN")}`,
                                    `H: ₹${highs[i].toLocaleString("en-IN")}`,
                                    `L: ₹${lows[i].toLocaleString("en-IN")}`,
                                    `C: ₹${closes[i].toLocaleString("en-IN")}`,
                                ];
                            }
                            return `Vol: ${item.raw.toLocaleString()}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid:  { color: GRID, drawBorder: false },
                    ticks: {
                        color: TEXT, maxRotation: 0, autoSkip: true,
                        maxTicksLimit: 12,
                        font: { size: 10, family: MONO }
                    }
                },
                y: {
                    position: "right",
                    grid:     { color: GRID, drawBorder: false },
                    ticks:    {
                        color: TEXT, font: { size: 10, family: MONO },
                        callback: v => "₹" + v.toLocaleString("en-IN")
                    }
                },
                yVol: {
                    position:   "left",
                    grid:       { display: false },
                    ticks:      { display: false },
                    max:        maxVol * 5,   // push volume bars to bottom 20%
                }
            }
        }
    });
}

// ── Risk gauge ring ───────────────────────────────────────────────────────────
function drawRiskRing(prob) {
    const canvas = document.getElementById("riskRing");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const sz  = 140;
    canvas.width  = sz * dpr; canvas.height = sz * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = sz+"px"; canvas.style.height = sz+"px";

    const cx=sz/2, cy=sz/2, r=58, lw=7, sa=-Math.PI/2;

    // Track
    ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI);
    ctx.lineWidth=lw; ctx.strokeStyle="rgba(48,54,61,0.8)"; ctx.stroke();

    // Arc colour
    const color = prob < 0.13 ? "#3fb950" : prob < 0.20 ? "#d29922" : "#f85149";
    let arcStyle = color;
    try {
        if (typeof ctx.createConicGradient === "function") {
            const g = ctx.createConicGradient(sa,cx,cy);
            g.addColorStop(0,   "#3fb950");
            g.addColorStop(0.35,"#d29922");
            g.addColorStop(0.65,"#f85149");
            g.addColorStop(1,   "#f85149");
            arcStyle = g;
        }
    } catch(e) {}

    const ea = sa + prob * 2 * Math.PI;
    ctx.beginPath(); ctx.arc(cx,cy,r,sa,ea);
    ctx.lineWidth=lw; ctx.strokeStyle=arcStyle; ctx.lineCap="round"; ctx.stroke();

    // Glow dot
    if (prob > 0.01) {
        const ex=cx+r*Math.cos(ea), ey=cy+r*Math.sin(ea);
        const g = ctx.createRadialGradient(ex,ey,0,ex,ey,10);
        const rgb = prob<0.13?"63,185,80":prob<0.20?"210,153,34":"248,81,73";
        g.addColorStop(0,`rgba(${rgb},.7)`); g.addColorStop(1,`rgba(${rgb},0)`);
        ctx.beginPath(); ctx.arc(ex,ey,10,0,2*Math.PI); ctx.fillStyle=g; ctx.fill();
    }
}

// ── Scan history chart ────────────────────────────────────────────────────────
function drawHistory() {
    const canvas = document.getElementById("historyChart");
    if (!canvas || !scanHistory.length) return;
    const ctx = canvas.getContext("2d");
    if (historyChart) { historyChart.destroy(); historyChart = null; }
    const grad = ctx.createLinearGradient(0,0,0,100);
    grad.addColorStop(0,"rgba(88,166,255,0.3)"); grad.addColorStop(1,"rgba(88,166,255,0)");
    historyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: scanHistory.map(s=>s.t),
            datasets:[{
                data:              scanHistory.map(s=>s.v),
                borderColor:       "#58a6ff",
                backgroundColor:   grad,
                borderWidth:       1.5,
                pointRadius:       2,
                pointBackgroundColor:"#58a6ff",
                fill:    true,
                tension: 0.3,
            }]
        },
        options:{
            responsive:true,maintainAspectRatio:false,
            plugins:{legend:{display:false},tooltip:{
                backgroundColor:"#161b22",titleColor:"#e6edf3",bodyColor:"#8b949e",
                cornerRadius:4,padding:8,
                titleFont:{family:"'JetBrains Mono',monospace",size:10},
                bodyFont: {family:"'JetBrains Mono',monospace",size:10},
                callbacks:{label:c=>`Risk: ${c.raw.toFixed(1)}%`}
            }},
            scales:{
                x:{display:false},
                y:{min:0,max:100,grid:{color:"rgba(48,54,61,0.5)"},
                   ticks:{color:"#484f58",font:{size:9}}}
            }
        }
    });
}

// ── Portfolio ─────────────────────────────────────────────────────────────────
async function runPortfolio() {
    const raw   = document.getElementById("portTickers")?.value.trim();
    const model = document.getElementById("modelSelect")?.value;
    const token = getToken();
    if (!raw) return showToast("Enter tickers");
    const tickers = raw.split(",").map(t=>t.trim()).filter(Boolean);
    const periodMap = {"1m":"5d","30m":"1mo","1d":"6mo","1wk":"2y","1mo":"5y"};
    const body_el = document.getElementById("portBody");
    body_el.innerHTML = `<tr><td colspan="4" style="text-align:center;padding:20px"><div class="spinner" style="margin:0 auto"></div></td></tr>`;
    try {
        const res  = await fetch(`${API}/api/portfolio`, {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({tickers,model,token,period:periodMap[currentInterval]||"6mo",interval:currentInterval,threshold:0.20})
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        body_el.innerHTML = data.results.map(r => {
            if (r.error) return `<tr><td style="font-weight:600">${r.ticker}</td><td colspan="3" style="color:var(--red)">${r.error}</td></tr>`;
            const color = bandHex(r.band), pct = r.risk_pct;
            return `<tr>
                <td style="font-weight:600;font-family:var(--mono)">${r.ticker.replace(".NS","")}</td>
                <td style="font-family:var(--mono)">₹${(r.latest_close||0).toLocaleString("en-IN")}</td>
                <td>
                    <span style="color:${color};font-family:var(--mono);font-weight:600">${pct.toFixed(1)}%</span>
                    <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:${Math.min(pct*5,100)}%;background:${color}"></div></div>
                </td>
                <td><span style="color:${color};font-size:11px;font-weight:600">${r.band}</span></td>
            </tr>`;
        }).join("");
    } catch(e) {
        body_el.innerHTML = `<tr><td colspan="4" style="color:var(--red);padding:16px">${e.message}</td></tr>`;
    }
}

// ── CSV ───────────────────────────────────────────────────────────────────────
function initCSV() {
    const zone  = document.getElementById("dropZone");
    const input = document.getElementById("csvInput");
    zone.addEventListener("click",    () => input.click());
    zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
    zone.addEventListener("drop", e => {
        e.preventDefault(); zone.classList.remove("drag");
        if (e.dataTransfer.files[0]) setCSVFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => { if (input.files[0]) setCSVFile(input.files[0]); });
    document.getElementById("btnCsv").addEventListener("click", runCSV);
}

function setCSVFile(f) {
    selectedFile = f;
    document.getElementById("dropZone").classList.add("has-file");
    document.getElementById("dropZone").innerHTML =
        `<div class="drop-icon">✅</div><div class="drop-text">${f.name}</div><div class="drop-hint">${(f.size/1024).toFixed(1)} KB</div>`;
}

async function runCSV() {
    if (!selectedFile) return showToast("Select a CSV file first");
    const model = document.getElementById("modelSelect")?.value;
    const res   = document.getElementById("csvResult");
    res.innerHTML = `<div style="display:flex;align-items:center;gap:8px;color:var(--text2)"><div class="spinner"></div>Analyzing…</div>`;
    try {
        const fd = new FormData();
        fd.append("file",selectedFile); fd.append("model",model); fd.append("threshold","0.20");
        const data = await (await fetch(`${API}/api/upload`,{method:"POST",body:fd})).json();
        if (data.error) throw new Error(data.error);
        const color = bandHex(data.band);
        res.innerHTML = `
            <div style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:14px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <span style="font-size:22px;font-weight:700;font-family:var(--mono);color:${color}">${data.risk_pct.toFixed(1)}%</span>
                    <span style="color:${color};font-size:12px;font-weight:600;background:rgba(255,255,255,.05);padding:3px 10px;border-radius:4px">${data.band}</span>
                </div>
                <div style="font-size:11px;color:var(--text2)">Rows: ${data.rows_loaded} · Model: ${(data.model||"").replace(".keras","")}</div>
            </div>`;
    } catch(e) {
        res.innerHTML = `<div style="color:var(--red);font-size:12px">${e.message}</div>`;
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function bandHex(b)  { return b==="STABLE"?"#3fb950":b==="ELEVATED"?"#d29922":"#f85149"; }
function bandClass(b){ return b==="STABLE"?"stable":b==="ELEVATED"?"elevated":"high"; }

function animateNum(el, target) {
    if (!el) return;
    const start = parseFloat(el.textContent)||0, diff=target-start, t0=performance.now();
    const tick = now => {
        const p=Math.min((now-t0)/600,1), e=1-Math.pow(1-p,3);
        el.textContent=(start+diff*e).toFixed(1);
        if(p<1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

function setText(id,txt) { const el=document.getElementById(id); if(el) el.textContent=txt; }

function showToast(msg) {
    const t=document.createElement("div"); t.className="toast"; t.textContent=msg;
    document.body.appendChild(t);
    setTimeout(()=>{t.style.opacity="0";t.style.transition="opacity .3s";setTimeout(()=>t.remove(),300);},4000);
}
