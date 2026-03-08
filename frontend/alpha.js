const alphaStatus = document.getElementById("alpha-status");
const alphaAsOf = document.getElementById("alpha-asof");
const alphaLastRun = document.getElementById("alpha-last-run");
const alphaSymbols = document.getElementById("alpha-symbols");
const alphaPoolMeta = document.getElementById("alpha-pool-meta");
const alphaSymbolList = document.getElementById("alpha-symbol-list");
const alphaDetailMeta = document.getElementById("alpha-detail-meta");
const alphaFactorRows = document.getElementById("alpha-factor-rows");
const alphaActiveSymbol = document.getElementById("alpha-active-symbol");
const alphaSymbolScore = document.getElementById("alpha-symbol-score");
const alphaSymbolAction = document.getElementById("alpha-symbol-action");
const alphaSymbolCoverage = document.getElementById("alpha-symbol-coverage");
const alphaSort = document.getElementById("alpha-sort");

let symbolList = [];
let symbolSummary = {};
let symbolDetails = {};
let activeSymbol = null;
let currentAsOf = null;
let sortDirection = "desc";

function formatScore(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return num.toFixed(2);
}

function formatValue(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  if (Math.abs(num) >= 1) return num.toFixed(3);
  return num.toFixed(5);
}

function formatSignal(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return num.toFixed(2);
}

function formatCategory(value) {
  if (!value) return "Custom";
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
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

function actionClass(action) {
  const normalized = String(action || "neutral").toLowerCase();
  if (normalized === "buy") return "gpt-action buy";
  if (normalized === "sell") return "gpt-action sell";
  return "gpt-action neutral";
}

function setStatus(state) {
  if (!alphaStatus) return;
  alphaStatus.textContent = state.toUpperCase();
  alphaStatus.classList.toggle("running", state === "running");
  alphaStatus.classList.toggle("error", state === "error");
}

function sortLabel() {
  return sortDirection === "desc" ? "Sort: BUY -> SELL" : "Sort: SELL -> BUY";
}

function sortSymbols(items) {
  const list = items.slice();
  list.sort((a, b) => {
    const scoreA = Number(a.score ?? -Infinity);
    const scoreB = Number(b.score ?? -Infinity);
    if (sortDirection === "asc") return scoreA - scoreB;
    return scoreB - scoreA;
  });
  return list;
}

function setActiveSymbol(symbol) {
  activeSymbol = symbol;
  fetchSymbolDetails(symbol);
  if (!alphaSymbolList) return;
  alphaSymbolList.querySelectorAll(".alpha-symbol-row").forEach((row) => {
    row.classList.toggle("active", row.dataset.symbol === symbol);
  });
}

function renderSymbolList(rows) {
  if (!alphaSymbolList) return;
  alphaSymbolList.innerHTML = "";
  if (!rows || rows.length === 0) {
    const row = document.createElement("div");
    row.className = "alpha-row alpha-empty";
    row.textContent = "No symbols loaded yet.";
    alphaSymbolList.appendChild(row);
    return;
  }

  sortSymbols(rows).forEach((item) => {
    const row = document.createElement("div");
    row.className = "alpha-row alpha-symbol-row";
    row.dataset.symbol = item.symbol;

    const action = document.createElement("span");
    action.className = actionClass(item.action);
    action.textContent = (item.action || "neutral").toUpperCase();

    // Add mini signal bar for quick visualization
    const score = item.score ?? 0;
    const absScore = Math.abs(score);
    const scoreWidth = Math.min(absScore * 50, 100);
    const scoreType = score > 0.2 ? 'buy' : score < -0.2 ? 'sell' : 'neutral';

    row.innerHTML = `
      <span class="alpha-symbol">${item.symbol}</span>
      <span class="alpha-category">${formatCategory(item.category)}</span>
      <span class="alpha-score">
        <div class="factor-signal-bar">
          <div class="signal-bar" style="height: 4px;">
            <div class="signal-bar-fill ${scoreType}" style="width: ${scoreWidth}%;"></div>
          </div>
          <span class="signal-value" style="min-width: 40px;">${formatScore(item.score)}</span>
        </div>
      </span>
      <span class="alpha-action-slot"></span>
      <span class="alpha-coverage">${item.coverage ?? "—"}</span>
    `;
    row.querySelector(".alpha-action-slot").appendChild(action);
    row.addEventListener("click", () => setActiveSymbol(item.symbol));
    if (item.symbol === activeSymbol) {
      row.classList.add("active");
    }
    alphaSymbolList.appendChild(row);
  });
}

function renderFactorTable() {
  if (!alphaFactorRows) return;
  alphaFactorRows.innerHTML = "";
  if (!activeSymbol) {
    const row = document.createElement("div");
    row.className = "alpha-row alpha-empty";
    row.textContent = "Select a symbol to view factor details.";
    alphaFactorRows.appendChild(row);
    return;
  }

  const detail = symbolDetails[activeSymbol];
  const summary = symbolSummary[activeSymbol];
  if (alphaDetailMeta) {
    const suffix = currentAsOf ? ` as of ${currentAsOf}` : "";
    alphaDetailMeta.textContent = `Factors for ${activeSymbol}${suffix}.`;
  }
  if (alphaActiveSymbol) {
    alphaActiveSymbol.textContent = activeSymbol;
  }
  if (alphaSymbolScore) {
    alphaSymbolScore.textContent = formatScore(detail?.score ?? summary?.score);
  }
  if (alphaSymbolCoverage) {
    alphaSymbolCoverage.textContent = `${detail?.coverage ?? summary?.coverage ?? 0} factors`;
  }
  if (alphaSymbolAction) {
    alphaSymbolAction.className = actionClass(detail?.action ?? summary?.action);
    alphaSymbolAction.textContent = (detail?.action || summary?.action || "neutral").toUpperCase();
  }

  if (!detail || !detail.factors) {
    const row = document.createElement("div");
    row.className = "alpha-row alpha-empty";
    row.textContent = "Loading factor details...";
    alphaFactorRows.appendChild(row);
    return;
  }

  detail.factors.forEach((factor) => {
    const row = document.createElement("div");
    row.className = "alpha-row alpha-factor-row";

    const action = document.createElement("span");
    action.className = actionClass(factor.action);
    action.textContent = (factor.action || "neutral").toUpperCase();

    // Create visual signal bar
    const signal = factor.signal ?? 0;
    const absSignal = Math.abs(signal);
    const signalWidth = Math.min(absSignal * 100, 100);
    const signalType = signal > 0.2 ? 'buy' : signal < -0.2 ? 'sell' : 'neutral';

    const signalBarHtml = `
      <div class="factor-signal-bar">
        <div class="signal-bar">
          <div class="signal-bar-fill ${signalType}" style="width: ${signalWidth}%;"></div>
        </div>
        <span class="signal-value">${formatSignal(factor.signal)}</span>
      </div>
    `;

    row.innerHTML = `
      <span class="alpha-factor">${factor.factor}</span>
      <span class="alpha-value">${formatValue(factor.value)}</span>
      <span class="alpha-signal-container">${signalBarHtml}</span>
      <span class="alpha-action-slot"></span>
      <span class="alpha-description">${factor.description || "—"}</span>
    `;
    row.querySelector(".alpha-action-slot").appendChild(action);
    alphaFactorRows.appendChild(row);
  });
}

async function fetchSymbolList() {
  setStatus("running");
  try {
    const response = await fetch("/api/factors/list");
    if (!response.ok) throw new Error("Failed to load factor data");
    const data = await response.json();
    if (alphaAsOf) alphaAsOf.textContent = data.as_of || "—";
    currentAsOf = data.as_of || null;
    if (alphaLastRun) alphaLastRun.textContent = formatDate(data.last_run);
    if (alphaSymbols) alphaSymbols.textContent = data.symbols?.length ?? "—";

    symbolList = data.symbols || [];
    symbolSummary = {};
    symbolList.forEach((item) => {
      symbolSummary[item.symbol] = item;
    });

    const sorted = sortSymbols(symbolList);
    if (!activeSymbol || !symbolSummary[activeSymbol]) {
      activeSymbol = sorted.length ? sorted[0].symbol : null;
    }

    renderSymbolList(symbolList);
    renderFactorTable();
    if (alphaPoolMeta) {
      alphaPoolMeta.textContent = `Loaded ${symbolList.length} symbols. ${sortLabel()}.`;
    }
    setStatus("idle");
    if (activeSymbol) {
      fetchSymbolDetails(activeSymbol);
    }
  } catch (error) {
    setStatus("error");
    if (alphaPoolMeta) alphaPoolMeta.textContent = "Unable to load factor data.";
  }
}

async function fetchSymbolDetails(symbol) {
  if (!symbol) return;
  if (symbolDetails[symbol]?.factors) {
    renderFactorTable();
    return;
  }
  renderFactorTable();
  try {
    const response = await fetch(`/api/factors/symbol?symbol=${encodeURIComponent(symbol)}`);
    if (!response.ok) throw new Error("Failed to load factor details");
    const data = await response.json();
    symbolDetails[symbol] = data;
    if (data.as_of) {
      currentAsOf = data.as_of;
      if (alphaAsOf) alphaAsOf.textContent = data.as_of;
    }
    renderFactorTable();
  } catch (error) {
    if (alphaFactorRows) {
      alphaFactorRows.innerHTML = "";
      const row = document.createElement("div");
      row.className = "alpha-row alpha-empty";
      row.textContent = "Unable to load factor details.";
      alphaFactorRows.appendChild(row);
    }
  }
}

if (alphaSort) {
  alphaSort.textContent = sortLabel();
  alphaSort.addEventListener("click", () => {
    sortDirection = sortDirection === "desc" ? "asc" : "desc";
    alphaSort.textContent = sortLabel();
    renderSymbolList(symbolList);
    if (alphaPoolMeta) {
      alphaPoolMeta.textContent = `Loaded ${symbolList.length} symbols. ${sortLabel()}.`;
    }
  });
}

fetchSymbolList();
