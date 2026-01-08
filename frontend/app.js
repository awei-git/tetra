const statusPill = document.getElementById("status-pill");
const lastRun = document.getElementById("last-run");
const lastError = document.getElementById("last-error");
const ingestMessage = document.getElementById("ingest-message");
const coverageRows = document.getElementById("coverage-rows");
const coverageMeta = document.getElementById("coverage-meta");
const chartCanvas = document.getElementById("ohlcv-chart");
const chartSymbol = document.getElementById("chart-symbol");
const chartRange = document.getElementById("chart-range");
const chartRows = document.getElementById("chart-rows");
const chartSources = document.getElementById("chart-sources");
const sentimentMeta = document.getElementById("sentiment-meta");
const sentimentSymbolRows = document.getElementById("sentiment-symbol-rows");
const sentimentMacroRows = document.getElementById("sentiment-macro-rows");
const eventsMeta = document.getElementById("events-meta");
const eventsRows = document.getElementById("events-rows");
const economicMeta = document.getElementById("economic-meta");
const economicRows = document.getElementById("economic-rows");

let coverageData = [];
let activeSymbol = null;
let lastSeries = [];

const fields = {
  assets: document.getElementById("data-assets"),
  ohlcv: document.getElementById("data-ohlcv"),
  events: document.getElementById("data-events"),
  economicSeries: document.getElementById("data-economic-series"),
  economicValues: document.getElementById("data-economic-values"),
  news: document.getElementById("data-news"),
  latestOhlcv: document.getElementById("latest-ohlcv"),
  latestEvents: document.getElementById("latest-events"),
  latestEconomic: document.getElementById("latest-economic"),
  latestNews: document.getElementById("latest-news"),
};

function formatValue(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  return value;
}

function formatDate(value) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toISOString().replace("T", " ").replace("Z", " UTC");
}

function formatShortDate(value) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toISOString().slice(0, 10);
}

function formatNumber(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  return value.toLocaleString();
}

function formatSentiment(value) {
  if (value === null || value === undefined) {
    return { text: "—", klass: "neutral" };
  }
  const rounded = Number(value).toFixed(3);
  if (value >= 0.05) {
    return { text: rounded, klass: "positive" };
  }
  if (value <= -0.05) {
    return { text: rounded, klass: "negative" };
  }
  return { text: rounded, klass: "neutral" };
}

function formatSources(sources) {
  if (!sources || sources.length === 0) {
    return "—";
  }
  return sources.join(", ");
}

function formatRange(start, end) {
  if (!start || !end) {
    return "—";
  }
  return `${formatShortDate(start)} → ${formatShortDate(end)}`;
}

function updateStatus(data) {
  const pipeline = data.pipeline || {};
  const status = pipeline.status || "idle";
  statusPill.textContent = status.toUpperCase();

  statusPill.style.background = status === "running"
    ? "rgba(255, 122, 89, 0.22)"
    : status === "failed"
      ? "rgba(255, 70, 70, 0.25)"
      : "rgba(63, 227, 196, 0.2)";

  lastRun.textContent = pipeline.last_run ? formatDate(pipeline.last_run) : "—";
  lastError.textContent = pipeline.last_error || "—";

  fields.assets.textContent = formatValue(data.counts?.assets);
  fields.ohlcv.textContent = formatValue(data.counts?.ohlcv);
  fields.events.textContent = formatValue(data.counts?.events);
  fields.economicSeries.textContent = formatValue(data.counts?.economic_series);
  fields.economicValues.textContent = formatValue(data.counts?.economic_values);
  fields.news.textContent = formatValue(data.counts?.news);

  fields.latestOhlcv.textContent = formatDate(data.latest?.ohlcv);
  fields.latestEvents.textContent = formatDate(data.latest?.events);
  fields.latestEconomic.textContent = formatDate(data.latest?.economic);
  fields.latestNews.textContent = formatDate(data.latest?.news);
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) {
      throw new Error("Failed to load status");
    }
    const data = await response.json();
    updateStatus(data);
  } catch (error) {
    ingestMessage.textContent = "Status unavailable.";
  }
}

async function runIngest(payload) {
  try {
    ingestMessage.textContent = "Starting ingestion...";
    const response = await fetch("/api/ingest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (response.status === 409) {
      ingestMessage.textContent = "Ingestion already running.";
      return;
    }

    if (!response.ok) {
      throw new Error("Failed to start ingestion");
    }

    const data = await response.json();
    ingestMessage.textContent = `Ingestion started: ${data.start} → ${data.end}`;
    await fetchStatus();
    await fetchCoverage();
  } catch (error) {
    ingestMessage.textContent = "Unable to start ingestion.";
  }
}

function renderCoverageTable(items) {
  coverageRows.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "coverage-row coverage-item";
    if (item.symbol === activeSymbol) {
      row.classList.add("active");
    }
    row.dataset.symbol = item.symbol;
    row.innerHTML = `
      <strong>${item.symbol}</strong>
      <span>${formatNumber(item.rows)}</span>
      <span>${formatShortDate(item.start)}</span>
      <span>${formatShortDate(item.end)}</span>
      <span>${formatSources(item.sources)}</span>
    `;
    row.addEventListener("click", () => {
      if (item.symbol !== activeSymbol) {
        activeSymbol = item.symbol;
        renderCoverageTable(items);
        fetchOhlcv(item.symbol, item);
      }
    });
    coverageRows.appendChild(row);
  });
}

function renderSentimentTable(target, rows, isMacro = false) {
  target.innerHTML = "";
  rows.forEach((row) => {
    const container = document.createElement("div");
    container.className = "sentiment-row";
    const sentiment = formatSentiment(row.avg_sentiment);
    const range = row.start && row.end
      ? `${formatShortDate(row.start)} → ${formatShortDate(row.end)}`
      : "—";
    container.innerHTML = `
      <strong>${isMacro ? row.topic : row.symbol}</strong>
      <span>${formatNumber(row.articles)}</span>
      <span class="sentiment-value ${sentiment.klass}">${sentiment.text}</span>
      <span>${range}</span>
    `;
    target.appendChild(container);
  });
}

async function fetchNewsSentiment() {
  try {
    const response = await fetch("/api/news/sentiment");
    if (!response.ok) {
      throw new Error("Failed to load sentiment");
    }
    const data = await response.json();
    const symbols = data.symbols || [];
    const macro = data.macro || [];
    const totalArticles = symbols.reduce((sum, row) => sum + (row.articles || 0), 0);
    sentimentMeta.textContent = `Articles scored: ${formatNumber(totalArticles)}`;
    renderSentimentTable(sentimentSymbolRows, symbols.slice(0, 20));
    renderSentimentTable(sentimentMacroRows, macro.slice(0, 20), true);
  } catch (error) {
    sentimentMeta.textContent = "Articles scored: —";
  }
}

function drawChart(series) {
  const ctx = chartCanvas.getContext("2d");
  if (!ctx) return;

  const width = chartCanvas.clientWidth;
  const height = chartCanvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  chartCanvas.width = width * ratio;
  chartCanvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(12, 19, 26, 0.6)";
  ctx.fillRect(0, 0, width, height);

  const filtered = (series || []).filter((point) => point.close !== null && point.close !== undefined);
  if (filtered.length === 0) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
    ctx.font = "14px JetBrains Mono, monospace";
    ctx.fillText("No data available", 16, 32);
    return;
  }

  const closes = filtered.map((point) => point.close);
  const minValue = Math.min(...closes);
  const maxValue = Math.max(...closes);
  const range = maxValue - minValue || 1;

  ctx.strokeStyle = "rgba(63, 227, 196, 0.8)";
  ctx.lineWidth = 2;
  ctx.beginPath();

  filtered.forEach((point, index) => {
    const x = (index / (filtered.length - 1)) * (width - 32) + 16;
    const normalized = (point.close - minValue) / range;
    const y = height - 20 - normalized * (height - 40);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(16, height - 20);
  ctx.lineTo(width - 16, height - 20);
  ctx.stroke();
}

async function fetchOhlcv(symbol, coverageItem) {
  if (!symbol) return;
  chartSymbol.textContent = symbol;
  chartRange.textContent = coverageItem
    ? `${formatShortDate(coverageItem.start)} → ${formatShortDate(coverageItem.end)}`
    : "Loading...";
  chartRows.textContent = coverageItem ? formatNumber(coverageItem.rows) : "—";
  chartSources.textContent = coverageItem ? formatSources(coverageItem.sources) : "—";

  try {
    const response = await fetch(`/api/market/ohlcv?symbol=${encodeURIComponent(symbol)}`);
    if (!response.ok) {
      throw new Error("Failed to load OHLCV");
    }
    const data = await response.json();
    lastSeries = data.series || [];
    drawChart(lastSeries);
  } catch (error) {
    lastSeries = [];
    drawChart([]);
  }
}

async function fetchCoverage() {
  try {
    const response = await fetch("/api/market/coverage");
    if (!response.ok) {
      throw new Error("Failed to load coverage");
    }
    const data = await response.json();
    coverageData = data.coverage || [];
    coverageMeta.textContent = `Symbols loaded: ${formatNumber(data.total_symbols || 0)}`;
    renderCoverageTable(coverageData);
    if (!activeSymbol && coverageData.length > 0) {
      activeSymbol = coverageData[0].symbol;
      renderCoverageTable(coverageData);
      fetchOhlcv(activeSymbol, coverageData[0]);
    }
  } catch (error) {
    coverageMeta.textContent = "Symbols loaded: —";
  }
}

function renderEventsTable(items) {
  eventsRows.innerHTML = "";
  if (!items || items.length === 0) {
    eventsRows.innerHTML = "<div class=\"events-row\"><span>—</span></div>";
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "events-row";
    row.innerHTML = `
      <span>${item.event_type || "—"}</span>
      <span>${formatNumber(item.events)}</span>
      <span>${formatNumber(item.symbols)}</span>
      <span>${formatRange(item.start, item.end)}</span>
      <span>${formatNumber(item.sources)}</span>
    `;
    eventsRows.appendChild(row);
  });
}

async function fetchEventsSummary() {
  try {
    const response = await fetch("/api/events/summary");
    if (!response.ok) {
      throw new Error("Failed to load events summary");
    }
    const data = await response.json();
    const totalTypes = formatNumber(data.total_types || 0);
    const totalEvents = formatNumber(data.total_events || 0);
    eventsMeta.textContent = `Event types: ${totalTypes} · Events: ${totalEvents}`;
    renderEventsTable(data.summary || []);
  } catch (error) {
    eventsMeta.textContent = "Event types: —";
  }
}

function renderEconomicTable(items) {
  economicRows.innerHTML = "";
  if (!items || items.length === 0) {
    economicRows.innerHTML = "<div class=\"economic-row\"><span>—</span></div>";
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "economic-row";
    const title = item.series_id || "—";
    const name = item.name || "";
    row.innerHTML = `
      <span><span class="row-title">${title}</span><span class="row-sub">${name}</span></span>
      <span>${item.frequency || "—"}</span>
      <span>${formatNumber(item.values)}</span>
      <span>${formatRange(item.start, item.end)}</span>
    `;
    economicRows.appendChild(row);
  });
}

async function fetchEconomicSummary() {
  try {
    const response = await fetch("/api/economic/summary");
    if (!response.ok) {
      throw new Error("Failed to load economic summary");
    }
    const data = await response.json();
    const totalSeries = formatNumber(data.total_series || 0);
    const totalValues = formatNumber(data.total_values || 0);
    economicMeta.textContent = `Series loaded: ${totalSeries} · Values: ${totalValues}`;
    renderEconomicTable(data.summary || []);
  } catch (error) {
    economicMeta.textContent = "Series loaded: —";
  }
}

const form = document.getElementById("refresh-form");
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const startDate = document.getElementById("start-date").value;
  const endDate = document.getElementById("end-date").value;
  const payload = {};
  if (startDate) payload.start_date = startDate;
  if (endDate) payload.end_date = endDate;
  runIngest(payload);
});

fetchStatus();
fetchCoverage();
fetchNewsSentiment();
fetchEventsSummary();
fetchEconomicSummary();
setInterval(fetchStatus, 30000);
setInterval(fetchCoverage, 60000);
setInterval(fetchNewsSentiment, 60000);
setInterval(fetchEventsSummary, 60000);
setInterval(fetchEconomicSummary, 60000);
window.addEventListener("resize", () => drawChart(lastSeries));
