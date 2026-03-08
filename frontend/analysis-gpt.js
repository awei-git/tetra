const recRows = document.getElementById("rec-rows");
const gptStatus = document.getElementById("gpt-status");
const gptLastRun = document.getElementById("gpt-last-run");
const gptSession = document.getElementById("gpt-session");
const gptRefresh = document.getElementById("gpt-refresh");
const gptError = document.getElementById("gpt-error");
const gptFinalError = document.getElementById("gpt-final-error");
const gptChallengeRows = document.getElementById("gpt-challenge-rows");
const gptChallengeLastRun = document.getElementById("gpt-challenge-last-run");
const gptChallengeRefresh = document.getElementById("gpt-challenge-refresh");
const gptChallengeError = document.getElementById("gpt-challenge-error");
const gptChallengeStatus = document.getElementById("gpt-challenge-status");

let lastChallengeRun = null;
let challengeTimer = null;

const gptRefreshLabel = gptRefresh
  ? (gptRefresh.querySelector(".btn-label") || gptRefresh).textContent
  : "Refresh";
const gptChallengeRefreshLabel = gptChallengeRefresh
  ? (gptChallengeRefresh.querySelector(".btn-label") || gptChallengeRefresh).textContent
  : "Refresh Challenge";

function preserveButtonWidth(button) {
  if (!button || button.dataset.minWidth) return;
  button.dataset.minWidth = String(button.offsetWidth);
  button.style.minWidth = `${button.dataset.minWidth}px`;
}

function setButtonLoading(button, label, isLoading) {
  if (!button) return;
  preserveButtonWidth(button);
  const labelNode = button.querySelector(".btn-label");
  if (isLoading) {
    button.disabled = true;
    button.classList.add("is-loading");
    button.setAttribute("aria-busy", "true");
    return;
  }
  button.disabled = false;
  button.classList.remove("is-loading");
  button.removeAttribute("aria-busy");
  if (labelNode) {
    labelNode.textContent = label;
  } else {
    button.textContent = label;
  }
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

function formatCategory(value) {
  if (!value) return "—";
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatAction(value) {
  if (!value) return "—";
  return String(value).toUpperCase();
}

function formatPrice(value) {
  if (value === null || value === undefined || value === "—") return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  if (Math.abs(num) >= 1) return num.toFixed(2);
  return num.toFixed(4);
}

function formatDelta(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  const formatted = Math.abs(num) >= 1 ? num.toFixed(2) : num.toFixed(3);
  return num > 0 ? `+${formatted}` : formatted;
}

function actionClass(value) {
  const action = String(value || "").toLowerCase();
  if (action === "buy") return "gpt-action buy";
  if (action === "sell") return "gpt-action sell";
  return "gpt-action neutral";
}

function changeClass(value) {
  const change = String(value || "").toLowerCase();
  if (change === "keep") return "gpt-change keep";
  if (change === "replace" || change === "exit") return "gpt-change replace";
  return "gpt-change adjust";
}

function createConfidenceGauge(confidence) {
  if (confidence === null || confidence === undefined) {
    return "<span>—</span>";
  }
  const pct = confidence * 100;
  const gaugeClass = pct >= 70 ? "high" : pct >= 40 ? "medium" : "low";
  return `
    <div class="confidence-gauge">
      <div class="gauge-bar">
        <div class="gauge-fill ${gaugeClass}" style="width: ${pct}%;"></div>
      </div>
      <span class="gauge-text">${pct.toFixed(0)}%</span>
    </div>
  `;
}

function setStatus(state) {
  if (!gptStatus) return;
  gptStatus.textContent = state.toUpperCase();
  gptStatus.classList.toggle("running", state === "running");
  gptStatus.classList.toggle("error", state === "error");
}

function setChallengeStatus(state) {
  if (!gptChallengeStatus) return;
  gptChallengeStatus.textContent = state.toUpperCase();
  gptChallengeStatus.classList.toggle("running", state === "running");
  gptChallengeStatus.classList.toggle("error", state === "error");
}

function renderRecommendations(rows) {
  if (!recRows) return;
  recRows.innerHTML = "";

  if (!rows || rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "rec-empty";
    empty.textContent = "No recommendations yet.";
    recRows.appendChild(empty);
    return;
  }

  rows.forEach((item) => {
    const horizon = item.horizon || (item.horizon_days ? `${item.horizon_days}d` : "—");
    const confidence = item.profit_prob ?? item.confidence;
    const signalText = item.signal_action
      ? `${formatAction(item.signal_action)}${item.score != null ? ` (${Number(item.score).toFixed(2)})` : ""}`
      : "—";
    const probText = [
      `base ${formatDelta(item.base_prob)}`,
      `signal ${formatDelta(item.signal_delta)}`,
      `review ${formatDelta(item.review_delta)}`,
      `challenge ${formatDelta(item.challenge_delta)}`,
      `= ${confidence != null ? (confidence * 100).toFixed(0) + "%" : "—"}`,
    ].join("  ·  ");

    // Summary row
    const row = document.createElement("div");
    row.className = "rec-row rec-row-grid";

    const symbolCell = document.createElement("div");
    symbolCell.className = "rec-symbol-cell";
    symbolCell.innerHTML = `<span class="rec-symbol">${item.symbol || "—"}</span><span class="rec-category">${formatCategory(item.category)}</span>`;

    const actionSpan = document.createElement("span");
    actionSpan.className = actionClass(item.final_action);
    actionSpan.textContent = formatAction(item.final_action);

    const confSpan = document.createElement("span");
    confSpan.innerHTML = createConfidenceGauge(confidence);

    const toggle = document.createElement("span");
    toggle.className = "rec-toggle";
    toggle.textContent = "▾";

    row.appendChild(symbolCell);
    row.appendChild(actionSpan);
    row.appendChild(confSpan);
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(item.entry) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(item.target) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(item.stop) }));

    const returnSpan = document.createElement("span");
    const entry = Number(item.entry);
    const target = Number(item.target);
    if (entry > 0 && target > 0) {
      const action = String(item.final_action || "").toLowerCase();
      const ret = action === "sell"
        ? (entry - target) / entry
        : (target - entry) / entry;
      returnSpan.textContent = (ret >= 0 ? "+" : "") + (ret * 100).toFixed(1) + "%";
      returnSpan.style.color = ret >= 0 ? "var(--buy-color, #4caf87)" : "var(--sell-color, #e05c5c)";
    } else {
      returnSpan.textContent = "—";
    }
    row.appendChild(returnSpan);

    row.appendChild(Object.assign(document.createElement("span"), { textContent: horizon }));
    row.appendChild(toggle);

    // Detail panel (hidden by default)
    const detail = document.createElement("div");
    detail.className = "rec-detail";
    detail.innerHTML = `
      <p class="rec-reason">${item.thesis || item.reason || "—"}</p>
      <div class="rec-tags">
        <span class="rec-tag">Signal: ${signalText}</span>
        <span class="rec-tag">Review: ${(item.review_verdict || "—").toUpperCase()}</span>
        <span class="rec-tag">Challenge: ${(item.challenge_change || "—").toUpperCase()}</span>
        ${item.providers && item.providers.length ? `<span class="rec-tag">Providers: ${item.providers.join(" · ")}</span>` : ""}
      </div>
      <p class="rec-prob">${probText}</p>
    `;

    row.addEventListener("click", () => {
      const isOpen = row.classList.toggle("expanded");
      detail.style.display = isOpen ? "block" : "none";
      toggle.textContent = isOpen ? "▴" : "▾";
    });

    recRows.appendChild(row);
    recRows.appendChild(detail);
  });
}

function renderChallengeRows(providers) {
  if (!gptChallengeRows) return;
  gptChallengeRows.innerHTML = "";
  const rows = [];

  providers.forEach((provider) => {
    const providerName = provider.provider || "—";
    if (provider.error) {
      rows.push({
        provider: providerName,
        category: "—",
        symbol: "—",
        last: "—",
        change: "—",
        action: "—",
        entry: "—",
        target: "—",
        stop: "—",
        notes: `Error: ${provider.error}`,
      });
      return;
    }

    const categories = provider.recommendations || {};
    Object.keys(categories).forEach((category) => {
      const items = categories[category] || [];
      items.forEach((item) => {
        rows.push({
          provider: providerName,
          category,
          symbol: item.symbol || "—",
          last: item.last_price ?? "—",
          change: item.change || "adjust",
          replacement: item.replaces || "—",
          action: item.action || "—",
          entry: item.entry || "—",
          target: item.target || "—",
          stop: item.stop || "—",
          notes: item.notes || item.thesis || "",
        });
      });
    });
  });

  if (rows.length === 0) {
    const row = document.createElement("div");
    row.className = "gpt-challenge-row";
    const span = document.createElement("span");
    span.textContent = "No GPT challenges yet. Try refresh.";
    row.appendChild(span);
    gptChallengeRows.appendChild(row);
    return;
  }

  rows.forEach((rowData) => {
    const row = document.createElement("div");
    row.className = "gpt-challenge-row";

    const changeSpan = document.createElement("span");
    changeSpan.className = changeClass(rowData.change);
    changeSpan.textContent = String(rowData.change || "adjust").toUpperCase();

    const actionSpan = document.createElement("span");
    actionSpan.className = actionClass(rowData.action);
    actionSpan.textContent = formatAction(rowData.action);

    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.provider }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatCategory(rowData.category) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.symbol }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(rowData.last) }));
    row.appendChild(changeSpan);
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.replacement }));
    row.appendChild(actionSpan);
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.entry }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.target }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.stop }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.notes }));

    gptChallengeRows.appendChild(row);
  });
}

async function fetchSummary() {
  try {
    const response = await fetch("/api/analysis/summary?min_factors=3&signal_threshold=0.1");
    if (!response.ok) throw new Error("Failed to load recommendations");
    const data = await response.json();
    if (gptFinalError) gptFinalError.textContent = "";
    if (gptLastRun) gptLastRun.textContent = formatDate(data.run_time);
    if (gptSession) gptSession.textContent = data.session || "—";
    renderRecommendations(data.final || []);
    setStatus("idle");
  } catch (error) {
    if (gptFinalError) gptFinalError.textContent = error.message || "Unable to load recommendations.";
    renderRecommendations([]);
    setStatus("error");
    setButtonLoading(gptRefresh, gptRefreshLabel, false);
  }
}

async function fetchChallenges(options = {}) {
  if (!gptChallengeRows) return null;
  try {
    const response = await fetch("/api/analysis/challenges");
    if (!response.ok) throw new Error("Failed to load GPT challenges");
    const data = await response.json();
    lastChallengeRun = data.run_time;
    if (gptChallengeError) gptChallengeError.textContent = data.last_error || "";
    if (gptChallengeLastRun) gptChallengeLastRun.textContent = formatDate(data.run_time);
    renderChallengeRows(data.providers || []);
    if (!options.keepStatus) {
      const status = data.status === "running" ? "running" : data.status === "failed" ? "error" : "idle";
      setChallengeStatus(status);
      setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, status === "running");
    }
    return data;
  } catch (error) {
    if (gptChallengeError) gptChallengeError.textContent = error.message || "Unable to load GPT challenges.";
    const row = document.createElement("div");
    row.className = "gpt-challenge-row";
    const span = document.createElement("span");
    span.textContent = "Unable to load GPT challenges.";
    row.appendChild(span);
    gptChallengeRows.innerHTML = "";
    gptChallengeRows.appendChild(row);
    setChallengeStatus("error");
    setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, false);
    return null;
  }
}

async function refreshSummary() {
  setStatus("running");
  setButtonLoading(gptRefresh, gptRefreshLabel, true);
  if (gptError) gptError.textContent = "";
  await fetchSummary();
  setButtonLoading(gptRefresh, gptRefreshLabel, false);
}

async function refreshChallenges() {
  if (!gptChallengeRows) return;
  try {
    setChallengeStatus("running");
    setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, true);
    if (gptChallengeError) gptChallengeError.textContent = "";
    const previousRun = lastChallengeRun;
    const response = await fetch("/api/analysis/challenges/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!response.ok) throw new Error("Failed to refresh GPT challenges");
    const payload = await response.json();
    if (payload.status === "running") {
      startChallengePolling(previousRun);
      return;
    }
    await fetchChallenges();
    setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, false);
  } catch (error) {
    setChallengeStatus("error");
    setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, false);
  }
}

function startChallengePolling(previousRun) {
  if (challengeTimer) clearTimeout(challengeTimer);
  let attempts = 0;
  const poll = async () => {
    attempts += 1;
    const data = await fetchChallenges({ keepStatus: true });
    if (data) {
      if (data.status === "failed") {
        setChallengeStatus("error");
        setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, false);
        return;
      }
      if (data.run_time && data.run_time !== previousRun && data.status !== "running") {
        setChallengeStatus("idle");
        setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, false);
        return;
      }
    }
    if (attempts >= 12) {
      setChallengeStatus("error");
      setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, false);
      if (gptChallengeError) gptChallengeError.textContent = "Refresh timed out. Try again in a moment.";
      return;
    }
    challengeTimer = setTimeout(poll, 5000);
  };
  challengeTimer = setTimeout(poll, 3000);
}

if (gptRefresh) {
  gptRefresh.addEventListener("click", refreshSummary);
}
if (gptChallengeRefresh) {
  gptChallengeRefresh.addEventListener("click", refreshChallenges);
}

fetchSummary();
fetchChallenges();
