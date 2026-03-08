const inferenceStatus = document.getElementById("inference-status");
const inferenceLastRun = document.getElementById("inference-last-run");
const inferenceAsOf = document.getElementById("inference-asof");

const signalAsOf = document.getElementById("inference-signal-asof");
const signalRows = document.getElementById("inference-signal-rows");

const eventAsOf = document.getElementById("inference-event-asof");
const eventRows = document.getElementById("inference-event-rows");

const polyAsOf = document.getElementById("inference-poly-asof");
const polyRows = document.getElementById("inference-poly-rows");
const polyMarkets = document.getElementById("inference-poly-markets");
const polyClosed = document.getElementById("inference-poly-closed");
const polyResolved = document.getElementById("inference-poly-resolved");
const polySpread = document.getElementById("inference-poly-spread");
const polyVolume = document.getElementById("inference-poly-volume");
const polyBrier = document.getElementById("inference-poly-brier");

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toISOString().slice(0, 10);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return `${(num * 100).toFixed(2)}%`;
}

function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return num.toFixed(decimals);
}

function setStatus(state) {
  if (!inferenceStatus) return;
  inferenceStatus.textContent = state.toUpperCase();
  inferenceStatus.classList.toggle("running", state === "running");
  inferenceStatus.classList.toggle("error", state === "error");
}

function renderSignalRows(rows) {
  if (!signalRows) return;
  signalRows.innerHTML = "";
  if (!rows.length) {
    const row = document.createElement("div");
    row.className = "inference-row";
    row.textContent = "No signal leaderboard rows.";
    signalRows.appendChild(row);
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "inference-row";
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.factor || "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: `${item.horizon_days}d` }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatNumber(item.avg_ic, 4) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatNumber(item.median_ic, 4) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPercent(item.hit_rate) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.days ?? "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.observations ?? "—" }));
    signalRows.appendChild(row);
  });
}

function renderEventRows(rows) {
  if (!eventRows) return;
  eventRows.innerHTML = "";
  if (!rows.length) {
    const row = document.createElement("div");
    row.className = "inference-row";
    row.textContent = "No event study rows.";
    eventRows.appendChild(row);
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "inference-row";
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.event_type || "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.window_days ?? "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPercent(item.avg_return) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPercent(item.median_return) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.observations ?? "—" }));
    eventRows.appendChild(row);
  });
}

function renderPolySummary(summary) {
  if (!summary) return;
  if (polyMarkets) polyMarkets.textContent = summary.markets ?? "—";
  if (polyClosed) polyClosed.textContent = summary.closed_markets ?? "—";
  if (polyResolved) polyResolved.textContent = summary.resolved_proxy ?? "—";
  if (polySpread) polySpread.textContent = formatNumber(summary.avg_spread, 4);
  if (polyVolume) polyVolume.textContent = formatNumber(summary.avg_volume, 2);
  if (polyBrier) polyBrier.textContent = formatNumber(summary.avg_brier, 4);
}

function renderPolyRows(rows) {
  if (!polyRows) return;
  polyRows.innerHTML = "";
  if (!rows.length) {
    const row = document.createElement("div");
    row.className = "inference-row";
    row.textContent = "No calibration bins available.";
    polyRows.appendChild(row);
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "inference-row";
    const binLabel = `${formatNumber(item.bin_low, 2)} - ${formatNumber(item.bin_high, 2)}`;
    row.appendChild(Object.assign(document.createElement("span"), { textContent: binLabel }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.count ?? "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatNumber(item.avg_pred, 4) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatNumber(item.proxy_accuracy, 4) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatNumber(item.avg_brier, 4) }));
    polyRows.appendChild(row);
  });
}

async function fetchSignals() {
  const response = await fetch("/api/inference/signal-leaderboard");
  if (!response.ok) throw new Error("Failed to load signal leaderboard");
  const data = await response.json();
  if (signalAsOf) signalAsOf.textContent = formatDate(data.as_of);
  renderSignalRows(data.rows || []);
  return data;
}

async function fetchEvents() {
  const response = await fetch("/api/inference/event-study");
  if (!response.ok) throw new Error("Failed to load event study");
  const data = await response.json();
  if (eventAsOf) eventAsOf.textContent = formatDate(data.as_of);
  renderEventRows(data.rows || []);
  return data;
}

async function fetchPolymarket() {
  const response = await fetch("/api/inference/polymarket");
  if (!response.ok) throw new Error("Failed to load Polymarket inference");
  const data = await response.json();
  if (polyAsOf) polyAsOf.textContent = formatDate(data.as_of);
  renderPolySummary(data.summary || {});
  renderPolyRows(data.bins || []);
  return data;
}

async function bootstrap() {
  setStatus("running");
  try {
    const [signals, events, poly] = await Promise.all([fetchSignals(), fetchEvents(), fetchPolymarket()]);
    if (inferenceLastRun) {
      inferenceLastRun.textContent = formatDate(signals.run_time || events.run_time || poly.run_time);
    }
    if (inferenceAsOf) {
      inferenceAsOf.textContent = formatDate(signals.as_of || events.as_of || poly.as_of);
    }
    setStatus("idle");
  } catch (error) {
    setStatus("error");
  }
}

bootstrap();
