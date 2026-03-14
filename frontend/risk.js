/* Risk Dashboard — fetches simulation data and renders panels */

const regimePill = document.getElementById("regime-pill");
const statusMeta = document.getElementById("status-meta");
const regimeDate = document.getElementById("regime-date");
const regimeBadgeRow = document.getElementById("regime-badge-row");
const regimeStates = document.getElementById("regime-states");
const regimeForecast = document.getElementById("regime-forecast");
const riskKpis = document.getElementById("risk-kpis");
const riskContributors = document.getElementById("risk-contributors");
const riskBreaches = document.getElementById("risk-breaches");
const riskMethodLabel = document.getElementById("risk-method-label");
const scenarioCount = document.getElementById("scenario-count");
const scenarioTbody = document.getElementById("scenario-tbody");
const scenarioDetail = document.getElementById("scenario-detail");
const corrCanvas = document.getElementById("corr-canvas");
const corrMeta = document.getElementById("corr-meta");
const volCanvas = document.getElementById("vol-canvas");
const volMeta = document.getElementById("vol-meta");
const runBtn = document.getElementById("run-simulation-btn");
const runStatus = document.getElementById("run-status");

function fmtPct(v, decimals = 1) {
  if (v === null || v === undefined) return "\u2014";
  return (v * 100).toFixed(decimals) + "%";
}

function fmtDollar(v) {
  if (v === null || v === undefined) return "\u2014";
  const sign = v < 0 ? "-" : "";
  return sign + "$" + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function fmtNum(v, d = 1) {
  if (v === null || v === undefined) return "\u2014";
  return Number(v).toFixed(d);
}

// ---- Regime Panel ----

async function loadRegime() {
  try {
    const res = await fetch("/api/simulation/regime");
    const data = await res.json();
    if (data.status === "no_data") {
      regimePill.textContent = "No data";
      regimePill.className = "status-pill";
      statusMeta.textContent = "Run simulation pipeline to populate.";
      return;
    }

    const regime = data.current_regime;
    regimePill.textContent = regime.toUpperCase();
    regimePill.className = "status-pill";
    regimeDate.textContent = data.as_of;
    statusMeta.textContent = `${data.n_observations} obs \u00b7 LL=${data.log_likelihood?.toFixed(0) || "\u2014"}`;

    // Badge row
    const probs = data.current_probs || {};
    regimeBadgeRow.innerHTML = "";
    const badge = document.createElement("div");
    badge.className = `regime-badge ${regime}`;
    badge.innerHTML = `<span class="dot"></span>${regime}`;
    regimeBadgeRow.appendChild(badge);

    const probSpan = document.createElement("span");
    probSpan.className = "regime-prob";
    const probParts = Object.entries(probs).map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`);
    probSpan.textContent = probParts.join(" \u00b7 ");
    regimeBadgeRow.appendChild(probSpan);

    // State cards
    const states = data.regime_states || [];
    regimeStates.innerHTML = "";
    states.forEach((s) => {
      const card = document.createElement("div");
      card.className = `regime-state-card ${s.label === regime ? "active" : ""}`;
      card.innerHTML = `
        <div class="label">${s.label}</div>
        <div class="value">${(s.mean_vol * 100).toFixed(1)}% vol</div>
        <div class="sub">${(s.frequency * 100).toFixed(0)}% of time \u00b7 ${s.duration_days.toFixed(0)}d avg</div>
      `;
      regimeStates.appendChild(card);
    });

    // Forecast
    const forecast = data.regime_forecast_5d || [];
    regimeForecast.innerHTML = "";
    forecast.forEach((day, i) => {
      const bar = document.createElement("div");
      bar.className = "forecast-bar";
      const entries = Object.entries(day);
      const dominant = entries.reduce((a, b) => (b[1] > a[1] ? b : a), ["", 0]);
      bar.innerHTML = `
        <div class="day-label">D+${i + 1}</div>
        <div style="color: ${regimeColor(dominant[0])}">${(dominant[1] * 100).toFixed(0)}% ${dominant[0]}</div>
      `;
      regimeForecast.appendChild(bar);
    });
  } catch (e) {
    regimePill.textContent = "Error";
    statusMeta.textContent = e.message;
  }
}

function regimeColor(label) {
  if (label === "calm") return "#3fe3c4";
  if (label === "stressed") return "#ffc759";
  if (label === "crisis") return "#ff5959";
  return "#999";
}

// ---- Risk Panel ----

async function loadRisk() {
  try {
    const res = await fetch("/api/simulation/risk");
    const data = await res.json();
    if (data.status === "no_data") {
      riskKpis.innerHTML = '<div class="risk-kpi"><div class="label">No data</div><div class="value">\u2014</div></div>';
      return;
    }

    // Prefer simulation method, fall back to parametric
    const method = data.methods.simulation ? "simulation" : "parametric";
    const r = data.methods[method];
    riskMethodLabel.textContent = method.charAt(0).toUpperCase() + method.slice(1);

    const kpis = [
      { label: "Ann. Vol", value: fmtPct(r.total_vol_ann), warn: r.total_vol_ann > 0.20 },
      { label: "VaR 95%", value: fmtDollar(r.var_95_1d), warn: false },
      { label: "CVaR 95%", value: fmtDollar(r.cvar_95_1d), warn: false },
      { label: "VaR 99%", value: fmtDollar(r.var_99_1d), warn: false },
      { label: "E[Max DD]", value: fmtPct(r.expected_max_drawdown), warn: r.expected_max_drawdown > 0.15, danger: r.expected_max_drawdown > 0.25 },
      { label: "Effective N", value: fmtNum(r.effective_n), warn: r.effective_n < 5 },
    ];

    riskKpis.innerHTML = kpis.map((k) => `
      <div class="risk-kpi">
        <div class="label">${k.label}</div>
        <div class="value ${k.danger ? "danger" : k.warn ? "warn" : ""}">${k.value}</div>
      </div>
    `).join("");

    // Risk contributors
    const comp = r.component_risk || {};
    const sorted = Object.entries(comp).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8);
    const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.001);

    riskContributors.innerHTML = sorted.map(([sym, v]) => {
      const pct = Math.abs(v) / maxAbs * 100;
      const neg = v < 0;
      return `
        <div class="risk-contrib-row">
          <span class="sym">${sym}</span>
          <div class="bar-wrap"><div class="bar ${neg ? "negative" : ""}" style="width:${pct.toFixed(1)}%"></div></div>
          <span class="pct">${fmtPct(v)}</span>
        </div>
      `;
    }).join("");

    // Breaches
    const breaches = r.risk_budget_breaches || {};
    const breachKeys = Object.keys(breaches);
    if (breachKeys.length > 0) {
      riskBreaches.innerHTML = breachKeys.map((k) =>
        `<span class="breach-tag">\u26a0 ${k}</span>`
      ).join("");
    } else {
      riskBreaches.innerHTML = '<span style="font-size:0.8rem;color:var(--accent-2)">No budget breaches</span>';
    }
  } catch (e) {
    riskKpis.innerHTML = `<div class="risk-kpi"><div class="value danger">${e.message}</div></div>`;
  }
}

// ---- Scenarios Panel ----

async function loadScenarios() {
  try {
    const res = await fetch("/api/simulation/scenarios");
    const data = await res.json();
    const scenarios = data.scenarios || [];
    scenarioCount.textContent = `${scenarios.length} scenarios \u00b7 ${data.as_of || ""}`;

    if (scenarios.length === 0) {
      scenarioTbody.innerHTML = '<tr><td colspan="5" style="color:var(--muted)">No scenario data. Run simulation pipeline.</td></tr>';
      return;
    }

    scenarioTbody.innerHTML = scenarios.map((s) => {
      const pnlClass = s.pnl_pct < 0 ? "pnl-neg" : "pnl-pos";
      return `
        <tr data-name="${s.name}">
          <td class="name-cell">${s.name}</td>
          <td class="${pnlClass}">${fmtDollar(s.pnl)}</td>
          <td class="${pnlClass}">${(s.pnl_pct * 100).toFixed(1)}%</td>
          <td>${s.worst_position || "\u2014"}</td>
          <td class="pnl-neg">${fmtDollar(s.worst_pnl)}</td>
        </tr>
      `;
    }).join("");

    // Click to show detail
    scenarioTbody.querySelectorAll("tr").forEach((tr) => {
      tr.style.cursor = "pointer";
      tr.addEventListener("click", () => {
        const name = tr.dataset.name;
        const sc = scenarios.find((s) => s.name === name);
        if (sc) {
          const pnls = sc.position_pnls || {};
          const sorted = Object.entries(pnls).sort((a, b) => a[1] - b[1]);
          const lines = sorted.slice(0, 5).map(([sym, pnl]) =>
            `${sym}: ${fmtDollar(pnl)}`
          );
          scenarioDetail.innerHTML = `
            <strong>${sc.name}</strong>: ${sc.description || ""}<br/>
            <span style="font-family:var(--mono);font-size:0.78rem">
              Top losses: ${lines.join(" \u00b7 ")}
            </span>
          `;
        }
      });
    });

    // Show first by default
    if (scenarios.length > 0) {
      const first = scenarios[0];
      scenarioDetail.innerHTML = `<strong>${first.name}</strong>: ${first.description || ""}`;
    }
  } catch (e) {
    scenarioTbody.innerHTML = `<tr><td colspan="5" style="color:#ff5959">${e.message}</td></tr>`;
  }
}

// ---- Correlation Heatmap ----

async function loadCorrelation() {
  try {
    const res = await fetch("/api/simulation/covariance?method=ledoit_wolf");
    const data = await res.json();
    if (data.status === "no_data") {
      corrMeta.textContent = "No data";
      return;
    }

    const symbols = data.symbols || [];
    const matrix = data.correlation_matrix || [];
    const n = symbols.length;

    corrMeta.textContent = `${n} assets \u00b7 ${data.n_observations} obs \u00b7 ${data.as_of}`;

    drawHeatmap(corrCanvas, symbols, matrix);
    drawVolBars(volCanvas, data.symbols, data.vols_ann);
    volMeta.textContent = `${data.method} \u00b7 ${data.as_of}`;
  } catch (e) {
    corrMeta.textContent = e.message;
  }
}

function drawHeatmap(canvas, symbols, matrix) {
  const n = symbols.length;
  if (n === 0) return;

  const labelW = 48;
  const cellSize = Math.min(Math.floor((500 - labelW) / n), 18);
  const totalSize = labelW + cellSize * n;

  const ratio = window.devicePixelRatio || 1;
  canvas.width = totalSize * ratio;
  canvas.height = totalSize * ratio;
  canvas.style.width = totalSize + "px";
  canvas.style.height = totalSize + "px";

  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, totalSize, totalSize);

  // Color scale: -1 (red) → 0 (dark) → 1 (cyan)
  function corrColor(v) {
    if (v >= 0) {
      const t = Math.min(v, 1);
      return `rgba(63, 227, 196, ${(t * 0.8 + 0.1).toFixed(2)})`;
    } else {
      const t = Math.min(Math.abs(v), 1);
      return `rgba(255, 89, 89, ${(t * 0.8 + 0.1).toFixed(2)})`;
    }
  }

  // Draw cells
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = matrix[i]?.[j] ?? 0;
      ctx.fillStyle = corrColor(val);
      ctx.fillRect(labelW + j * cellSize, labelW + i * cellSize, cellSize - 1, cellSize - 1);
    }
  }

  // Labels
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.font = `${Math.min(cellSize - 2, 9)}px JetBrains Mono`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    ctx.fillText(symbols[i], labelW - 4, labelW + i * cellSize + cellSize / 2);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let j = 0; j < n; j++) {
    ctx.save();
    ctx.translate(labelW + j * cellSize + cellSize / 2, labelW - 4);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(symbols[j], 0, 0);
    ctx.restore();
  }
}

function drawVolBars(canvas, symbols, volsObj) {
  if (!symbols || !volsObj) return;

  const vols = symbols.map((s) => volsObj[s] || 0);
  const n = symbols.length;

  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const marginLeft = 10;
  const marginBottom = 40;
  const marginTop = 10;
  const barW = Math.min(Math.floor((width - marginLeft) / n) - 2, 24);
  const chartH = height - marginBottom - marginTop;
  const maxVol = Math.max(...vols, 0.01);

  for (let i = 0; i < n; i++) {
    const v = vols[i];
    const barH = (v / maxVol) * chartH;
    const x = marginLeft + i * (barW + 2);
    const y = marginTop + chartH - barH;

    // Color by vol level
    const pct = v / maxVol;
    if (pct > 0.7) {
      ctx.fillStyle = "rgba(255, 89, 89, 0.7)";
    } else if (pct > 0.4) {
      ctx.fillStyle = "rgba(255, 199, 89, 0.7)";
    } else {
      ctx.fillStyle = "rgba(63, 227, 196, 0.6)";
    }

    ctx.fillRect(x, y, barW, barH);

    // Label
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = `${Math.min(barW - 1, 8)}px JetBrains Mono`;
    ctx.textAlign = "center";
    ctx.translate(x + barW / 2, marginTop + chartH + 4);
    ctx.rotate(Math.PI / 3);
    ctx.fillText(symbols[i], 0, 0);
    ctx.restore();

    // Vol value on top
    if (barW >= 10) {
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = `${Math.min(barW - 2, 7)}px JetBrains Mono`;
      ctx.textAlign = "center";
      ctx.fillText((v * 100).toFixed(0) + "%", x + barW / 2, y - 3);
    }
  }
}

// ---- Run Simulation ----

if (runBtn) {
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    runStatus.textContent = "Running simulation pipeline...";
    try {
      const res = await fetch("/api/simulation/run", { method: "POST" });
      const data = await res.json();
      if (data.status === "error") {
        runStatus.textContent = `Error: ${data.error}`;
      } else {
        runStatus.textContent = `Complete in ${data.elapsed_seconds || "?"}s. Refreshing...`;
        setTimeout(() => {
          loadAll();
          runStatus.textContent = "Dashboard refreshed.";
        }, 500);
      }
    } catch (e) {
      runStatus.textContent = `Failed: ${e.message}`;
    } finally {
      runBtn.disabled = false;
    }
  });
}

// ---- Load All ----

function loadAll() {
  loadRegime();
  loadRisk();
  loadScenarios();
  loadCorrelation();
}

loadAll();
