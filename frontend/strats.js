const simForm = document.getElementById("sim-form");
const simSymbol = document.getElementById("sim-symbol");
const simMethod = document.getElementById("sim-method");
const simStress = document.getElementById("sim-stress");
const simMode = document.getElementById("sim-mode");
const simHorizon = document.getElementById("sim-horizon");
const simPaths = document.getElementById("sim-paths");
const simMessage = document.getElementById("sim-message");
const simChart = document.getElementById("sim-chart");
const simSummary = document.getElementById("sim-summary");
const simTitle = document.getElementById("sim-title");
const simMeta = document.getElementById("sim-meta");
const stressField = document.querySelector(".sim-stress-field");
const modeField = document.querySelector(".sim-mode-field");

let stressOptions = [];

function formatPrice(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return value;
  return Math.abs(num) >= 1 ? num.toFixed(2) : num.toFixed(4);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return `${(num * 100).toFixed(1)}%`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZoneName: "short",
  });
  const parts = formatter.formatToParts(date);
  const lookup = {};
  parts.forEach((part) => {
    lookup[part.type] = part.value;
  });
  const tz = lookup.timeZoneName || "EST";
  return `${lookup.year}-${lookup.month}-${lookup.day} ${lookup.hour}:${lookup.minute} ${tz}`;
}

function setMessage(text) {
  if (simMessage) {
    simMessage.textContent = text;
  }
}

function updateMethodFields() {
  const method = simMethod ? simMethod.value : "historical";
  if (stressField) {
    stressField.style.display = method === "stress" ? "grid" : "none";
  }
  if (modeField) {
    modeField.style.display = method === "historical" ? "grid" : "none";
  }
}

function setStressOptions(options, selectedKey) {
  if (!simStress) return;
  simStress.innerHTML = "";
  options.forEach((option) => {
    const el = document.createElement("option");
    el.value = option.key;
    el.textContent = option.label;
    if (option.key === selectedKey) {
      el.selected = true;
    }
    simStress.appendChild(el);
  });
}

function drawSimulation(paths) {
  if (!simChart) return;
  const ctx = simChart.getContext("2d");
  if (!ctx) return;
  const width = simChart.clientWidth;
  const height = simChart.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  simChart.width = width * ratio;
  simChart.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(12, 19, 26, 0.6)";
  ctx.fillRect(0, 0, width, height);

  if (!paths || paths.length === 0) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
    ctx.font = "14px JetBrains Mono, monospace";
    ctx.fillText("No simulation data", 16, 32);
    return;
  }

  const allPrices = paths.flatMap((path) => path.prices || []);
  const minValue = Math.min(...allPrices);
  const maxValue = Math.max(...allPrices);
  const range = maxValue - minValue || 1;

  const endPrices = paths.map((path) => path.prices?.[path.prices.length - 1] ?? 0);
  const sortedEnds = [...endPrices].sort((a, b) => a - b);
  const medianEnd = sortedEnds[Math.floor(sortedEnds.length / 2)];
  let medianIndex = 0;
  let closest = Infinity;
  endPrices.forEach((value, idx) => {
    const diff = Math.abs(value - medianEnd);
    if (diff < closest) {
      closest = diff;
      medianIndex = idx;
    }
  });

  paths.forEach((path, index) => {
    const prices = path.prices || [];
    if (prices.length === 0) return;
    ctx.strokeStyle = index === medianIndex
      ? "rgba(255, 199, 89, 0.9)"
      : "rgba(63, 227, 196, 0.25)";
    ctx.lineWidth = index === medianIndex ? 2.4 : 1.1;
    ctx.beginPath();
    prices.forEach((value, step) => {
      const x = (step / (prices.length - 1)) * (width - 32) + 16;
      const normalized = (value - minValue) / range;
      const y = height - 20 - normalized * (height - 40);
      if (step === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });

  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(16, height - 20);
  ctx.lineTo(width - 16, height - 20);
  ctx.stroke();
}

function renderSummary(data) {
  if (!simSummary) return;
  const summary = data.summary || {};
  simSummary.innerHTML = `
    <div class="sim-summary-grid">
      <div>
        <span>Start</span>
        <strong>${formatPrice(summary.start_price)}</strong>
      </div>
      <div>
        <span>Median end</span>
        <strong>${formatPrice(summary.median_end)}</strong>
      </div>
      <div>
        <span>P05 / P95</span>
        <strong>${formatPrice(summary.p05_end)} → ${formatPrice(summary.p95_end)}</strong>
      </div>
      <div>
        <span>Mean end</span>
        <strong>${formatPrice(summary.mean_end)}</strong>
      </div>
      <div>
        <span>Median return</span>
        <strong>${formatPercent(summary.median_return)}</strong>
      </div>
      <div>
        <span>P05 / P95 return</span>
        <strong>${formatPercent(summary.p05_return)} → ${formatPercent(summary.p95_return)}</strong>
      </div>
    </div>
  `;
}

async function runSimulation() {
  if (!simSymbol) return;
  const symbol = simSymbol.value.trim().toUpperCase();
  if (!symbol) {
    setMessage("Symbol is required.");
    return;
  }
  const method = simMethod ? simMethod.value : "historical";
  const horizon = simHorizon ? Number(simHorizon.value) : 252;
  const paths = simPaths ? Number(simPaths.value) : 12;
  const params = new URLSearchParams();
  params.set("symbol", symbol);
  params.set("method", method);
  params.set("horizon", String(horizon));
  params.set("paths", String(paths));
  if (method === "stress" && simStress) {
    params.set("stress", simStress.value);
  }
  if (method === "historical" && simMode) {
    params.set("mode", simMode.value);
  }

  setMessage("Running simulation...");
  try {
    const response = await fetch(`/api/market/simulations?${params.toString()}`);
    if (!response.ok) {
      throw new Error("Simulation request failed");
    }
    const data = await response.json();
    stressOptions = data.available_stress || stressOptions;
    if (stressOptions.length > 0) {
      setStressOptions(stressOptions, data.stress?.key || simStress?.value);
    }
    simTitle.textContent = `${data.symbol} · ${data.method.replace(/_/g, " ")}`;
    const metaBits = [
      `As of ${formatDate(data.as_of)}`,
      `${data.horizon} days`,
      `${data.paths} paths`,
    ];
    if (data.method === "stress" && data.stress) {
      metaBits.push(data.stress.label);
    }
    simMeta.textContent = metaBits.join(" · ");
    drawSimulation(data.paths_data || []);
    renderSummary(data);
    setMessage("Simulation complete.");
  } catch (error) {
    setMessage("Unable to run simulation.");
    drawSimulation([]);
  }
}

if (simMethod) {
  simMethod.addEventListener("change", () => {
    updateMethodFields();
  });
}

if (simForm) {
  simForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runSimulation();
  });
}

updateMethodFields();
runSimulation();
