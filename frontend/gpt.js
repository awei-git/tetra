const gptConsensusRows = document.getElementById("gpt-consensus-rows");
const gptStatus = document.getElementById("gpt-status");
const gptLastRun = document.getElementById("gpt-last-run");
const gptSession = document.getElementById("gpt-session");
const gptLastRunInline = document.getElementById("gpt-last-run-inline");
const gptSessionInline = document.getElementById("gpt-session-inline");
const gptRefresh = document.getElementById("gpt-refresh");
const gptError = document.getElementById("gpt-error");
let lastRunTime = null;
let refreshTimer = null;
const gptChallengeRows = document.getElementById("gpt-challenge-rows");
const gptChallengeLastRun = document.getElementById("gpt-challenge-last-run");
const gptChallengeSession = document.getElementById("gpt-challenge-session");
const gptChallengeRefresh = document.getElementById("gpt-challenge-refresh");
const gptChallengeError = document.getElementById("gpt-challenge-error");
const gptChallengeStatus = document.getElementById("gpt-challenge-status");
let lastChallengeRun = null;
let challengeTimer = null;
const gptFactorRows = document.getElementById("gpt-factor-rows");
const gptFactorLastRun = document.getElementById("gpt-factor-last-run");
const gptFactorSession = document.getElementById("gpt-factor-session");
const gptFactorError = document.getElementById("gpt-factor-error");
const gptFactorStatus = document.getElementById("gpt-factor-status");
let lastFactorRun = null;
const gptFinalRows = document.getElementById("gpt-final-rows");
const gptFinalStatus = document.getElementById("gpt-final-status");
const gptFinalLastRun = document.getElementById("gpt-final-last-run");
const gptFinalAsOf = document.getElementById("gpt-final-asof");
const gptFinalError = document.getElementById("gpt-final-error");
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

function verdictClass(value) {
  const verdict = String(value || "").toLowerCase();
  if (verdict === "approve") return "gpt-verdict approve";
  if (verdict === "reject") return "gpt-verdict reject";
  return "gpt-verdict watch";
}

function setStatus(state) {
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

function setFactorStatus(state) {
  if (!gptFactorStatus) return;
  gptFactorStatus.textContent = state.toUpperCase();
  gptFactorStatus.classList.toggle("running", state === "running");
  gptFactorStatus.classList.toggle("error", state === "error");
}

function setFinalStatus(state) {
  if (!gptFinalStatus) return;
  gptFinalStatus.textContent = state.toUpperCase();
  gptFinalStatus.classList.toggle("running", state === "running");
  gptFinalStatus.classList.toggle("error", state === "error");
}

function renderEmpty(message) {
  if (!gptConsensusRows) return;
  gptConsensusRows.innerHTML = "";
  const row = document.createElement("div");
  row.className = "gpt-consensus-row";
  const span = document.createElement("span");
  span.textContent = message || "No consensus rows yet.";
  row.appendChild(span);
  gptConsensusRows.appendChild(row);
}

function renderConsensusRows(rows, byCategory) {
  if (!gptConsensusRows) return;
  gptConsensusRows.innerHTML = "";

  const grouped = byCategory && Object.keys(byCategory).length > 0;
  if ((!rows || rows.length === 0) && !grouped) {
    renderEmpty("No consensus rows yet. Try refresh.");
    return;
  }

  const categoryOrder = ["large_cap", "growth", "etf", "crypto"];
  const renderRow = (rowData) => {
    const row = document.createElement("div");
    row.className = "gpt-consensus-row";
    row.title = rowData.thesis || "No reasoning provided.";

    const actionSpan = document.createElement("span");
    actionSpan.className = actionClass(rowData.action);
    actionSpan.textContent = formatAction(rowData.action);

    const expReturn = rowData.expected_return;
    const rr = rowData.reward_risk;
    const confidence = rowData.confidence;

    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.rank || "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatCategory(rowData.category) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.symbol || "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(rowData.last_price) }));
    row.appendChild(actionSpan);
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(rowData.entry) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(rowData.target) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(rowData.stop) }));
    row.appendChild(
      Object.assign(document.createElement("span"), {
        textContent: expReturn === null || expReturn === undefined ? "—" : `${(expReturn * 100).toFixed(1)}%`,
      }),
    );
    row.appendChild(
      Object.assign(document.createElement("span"), {
        textContent: rr === null || rr === undefined ? "—" : Number(rr).toFixed(2),
      }),
    );
    row.appendChild(
      Object.assign(document.createElement("span"), {
        textContent:
          confidence === null || confidence === undefined ? "—" : `${(confidence * 100).toFixed(1)}%`,
      }),
    );
    row.appendChild(Object.assign(document.createElement("span"), { textContent: (rowData.providers || []).join(", ") }));

    gptConsensusRows.appendChild(row);
  };

  if (grouped) {
    categoryOrder.forEach((category) => {
      const groupRow = document.createElement("div");
      groupRow.className = "gpt-consensus-group";
      const span = document.createElement("span");
      span.textContent = formatCategory(category);
      groupRow.appendChild(span);
      gptConsensusRows.appendChild(groupRow);

      const items = byCategory[category] || [];
      if (items.length === 0) {
        const emptyRow = document.createElement("div");
        emptyRow.className = "gpt-consensus-row";
        const emptySpan = document.createElement("span");
        emptySpan.textContent = "No consensus yet.";
        emptyRow.appendChild(emptySpan);
        gptConsensusRows.appendChild(emptyRow);
        return;
      }
      items.forEach(renderRow);
    });
    return;
  }

  rows.forEach(renderRow);
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
        horizon: "—",
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
          horizon: item.horizon || "—",
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
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.horizon }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.notes }));

    gptChallengeRows.appendChild(row);
  });
}

function renderFactorRows(consensus) {
  if (!gptFactorRows) return;
  gptFactorRows.innerHTML = "";
  const rows = consensus || [];
  if (rows.length === 0) {
    const row = document.createElement("div");
    row.className = "gpt-factor-row";
    const span = document.createElement("span");
    span.textContent = "No factor reviews yet.";
    row.appendChild(span);
    gptFactorRows.appendChild(row);
    return;
  }

  rows.forEach((rowData) => {
    const row = document.createElement("div");
    row.className = "gpt-factor-row";
    const drivers = (rowData.drivers || [])
      .map((driver) => `${driver.factor}:${driver.signal === null || driver.signal === undefined ? "—" : Number(driver.signal).toFixed(2)}`)
      .join(" | ");
    if (drivers) {
      row.title = drivers;
    }

    const factorAction = document.createElement("span");
    factorAction.className = actionClass(rowData.factor_action);
    factorAction.textContent = formatAction(rowData.factor_action);

    const verdict = document.createElement("span");
    verdict.className = verdictClass(rowData.verdict);
    verdict.textContent = String(rowData.verdict || "watch").toUpperCase();

    const action = document.createElement("span");
    action.className = actionClass(rowData.action);
    action.textContent = formatAction(rowData.action);

    const confidence = rowData.confidence;
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatCategory(rowData.category) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.symbol || "—" }));
    row.appendChild(factorAction);
    const scoreValue = Number(rowData.factor_score);
    row.appendChild(
      Object.assign(document.createElement("span"), {
        textContent:
          rowData.factor_score === null || rowData.factor_score === undefined || Number.isNaN(scoreValue)
            ? "—"
            : scoreValue.toFixed(2),
      }),
    );
    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatPrice(rowData.last_price) }));
    row.appendChild(verdict);
    row.appendChild(action);
    row.appendChild(
      Object.assign(document.createElement("span"), {
        textContent:
          confidence === null || confidence === undefined ? "—" : `${(confidence * 100).toFixed(1)}%`,
      }),
    );
    row.appendChild(Object.assign(document.createElement("span"), { textContent: (rowData.providers || []).join(", ") }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.replacement || "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: rowData.notes || "—" }));

    gptFactorRows.appendChild(row);
  });
}

function renderFinalRows(rows) {
  if (!gptFinalRows) return;
  gptFinalRows.innerHTML = "";
  const items = rows || [];
  if (items.length === 0) {
    const row = document.createElement("div");
    row.className = "gpt-final-row";
    const span = document.createElement("span");
    span.textContent = "No consolidated verdicts yet.";
    row.appendChild(span);
    gptFinalRows.appendChild(row);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "gpt-final-row";

    const verdict = document.createElement("span");
    verdict.className = actionClass(item.final_action);
    verdict.textContent = formatAction(item.final_action);

    const confidence = item.confidence;
    const scoreValue = Number(item.score);

    row.appendChild(Object.assign(document.createElement("span"), { textContent: formatCategory(item.category) }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.symbol || "—" }));
    row.appendChild(verdict);
    row.appendChild(
      Object.assign(document.createElement("span"), {
        textContent:
          confidence === null || confidence === undefined ? "—" : `${(confidence * 100).toFixed(1)}%`,
      }),
    );
    row.appendChild(
      Object.assign(document.createElement("span"), {
        textContent:
          item.score === null || item.score === undefined || Number.isNaN(scoreValue) ? "—" : scoreValue.toFixed(2),
      }),
    );
    row.appendChild(Object.assign(document.createElement("span"), { textContent: (item.review_verdict || "—").toUpperCase() }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: (item.challenge_change || "—").toUpperCase() }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: (item.providers || []).join(", ") }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.replacement || "—" }));
    row.appendChild(Object.assign(document.createElement("span"), { textContent: item.notes || "—" }));

    gptFinalRows.appendChild(row);
  });
}

async function fetchRecommendations(options = {}) {
  try {
    const response = await fetch("/api/gpt/consensus");
    if (!response.ok) {
      throw new Error("Failed to load GPT consensus");
    }
    const data = await response.json();
    lastRunTime = data.run_time;
    if (gptError) {
      gptError.textContent = data.last_error || "";
    }
    gptLastRun.textContent = formatDate(data.run_time);
    gptSession.textContent = data.session || data.last_session || "—";
    if (gptLastRunInline) {
      gptLastRunInline.textContent = formatDate(data.run_time);
    }
    if (gptSessionInline) {
      gptSessionInline.textContent = data.session || data.last_session || "—";
    }
    renderConsensusRows(data.consensus || [], data.by_category || {});
    if (!options.keepStatus) {
      const status = data.status === "running" ? "running" : data.status === "failed" ? "error" : "idle";
      setStatus(status);
      setButtonLoading(gptRefresh, gptRefreshLabel, status === "running");
    }
    return data;
  } catch (error) {
    if (gptError) {
      gptError.textContent = error.message || "Unable to load GPT data.";
    }
    renderEmpty("Unable to load GPT data.");
    setStatus("error");
    setButtonLoading(gptRefresh, gptRefreshLabel, false);
    return null;
  }
}

async function fetchChallenges(options = {}) {
  if (!gptChallengeRows) return null;
  try {
    const response = await fetch("/api/gpt/challenges");
    if (!response.ok) {
      throw new Error("Failed to load GPT challenges");
    }
    const data = await response.json();
    lastChallengeRun = data.run_time;
    if (gptChallengeError) {
      gptChallengeError.textContent = data.last_error || "";
    }
    if (gptChallengeLastRun) {
      gptChallengeLastRun.textContent = formatDate(data.run_time);
    }
    if (gptChallengeSession) {
      gptChallengeSession.textContent = data.session || data.last_session || "—";
    }
    renderChallengeRows(data.providers || []);
    if (!options.keepStatus) {
      const status = data.status === "running" ? "running" : data.status === "failed" ? "error" : "idle";
      setChallengeStatus(status);
      setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, status === "running");
    }
    return data;
  } catch (error) {
    if (gptChallengeError) {
      gptChallengeError.textContent = error.message || "Unable to load GPT challenges.";
    }
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

async function fetchFactorReviews(options = {}) {
  if (!gptFactorRows) return null;
  try {
    const response = await fetch("/api/gpt/factor-reviews");
    if (!response.ok) {
      throw new Error("Failed to load factor reviews");
    }
    const data = await response.json();
    lastFactorRun = data.run_time;
    if (gptFactorError) {
      gptFactorError.textContent = data.last_error || "";
    }
    if (gptFactorLastRun) {
      gptFactorLastRun.textContent = formatDate(data.run_time);
    }
    if (gptFactorSession) {
      gptFactorSession.textContent = data.session || data.last_session || "—";
    }
    renderFactorRows(data.consensus || []);
    if (!options.keepStatus) {
      const status = data.status === "running" ? "running" : data.status === "failed" ? "error" : "idle";
      setFactorStatus(status);
    }
    return data;
  } catch (error) {
    if (gptFactorError) {
      gptFactorError.textContent = error.message || "Unable to load factor reviews.";
    }
    if (gptFactorRows) {
      const row = document.createElement("div");
      row.className = "gpt-factor-row";
      const span = document.createElement("span");
      span.textContent = "Unable to load factor reviews.";
      row.appendChild(span);
      gptFactorRows.innerHTML = "";
      gptFactorRows.appendChild(row);
    }
    setFactorStatus("error");
    return null;
  }
}

async function fetchSummary(options = {}) {
  if (!gptFinalRows) return null;
  try {
    const response = await fetch("/api/gpt/summary");
    if (!response.ok) {
      throw new Error("Failed to load consolidated verdicts");
    }
    const data = await response.json();
    if (gptFinalError) {
      gptFinalError.textContent = "";
    }
    if (gptFinalLastRun) {
      gptFinalLastRun.textContent = formatDate(data.run_time);
    }
    if (gptFinalAsOf) {
      gptFinalAsOf.textContent = data.as_of || "—";
    }
    renderFinalRows(data.final || []);
    if (!options.keepStatus) {
      setFinalStatus("idle");
    }
    return data;
  } catch (error) {
    if (gptFinalError) {
      gptFinalError.textContent = error.message || "Unable to load consolidated verdicts.";
    }
    if (gptFinalRows) {
      const row = document.createElement("div");
      row.className = "gpt-final-row";
      const span = document.createElement("span");
      span.textContent = "Unable to load consolidated verdicts.";
      row.appendChild(span);
      gptFinalRows.innerHTML = "";
      gptFinalRows.appendChild(row);
    }
    setFinalStatus("error");
    return null;
  }
}

async function refreshRecommendations() {
  try {
    setStatus("running");
    setButtonLoading(gptRefresh, gptRefreshLabel, true);
    if (gptError) {
      gptError.textContent = "";
    }
    const previousRun = lastRunTime;
    const response = await fetch("/api/gpt/recommendations/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      throw new Error("Failed to refresh GPT recommendations");
    }
    const payload = await response.json();
    if (payload.status === "running") {
      startPolling(previousRun);
      return;
    }
    await fetchRecommendations();
    setButtonLoading(gptRefresh, gptRefreshLabel, false);
  } catch (error) {
    setStatus("error");
    setButtonLoading(gptRefresh, gptRefreshLabel, false);
  }
}

async function refreshChallenges() {
  if (!gptChallengeRows) return;
  try {
    setChallengeStatus("running");
    setButtonLoading(gptChallengeRefresh, gptChallengeRefreshLabel, true);
    if (gptChallengeError) {
      gptChallengeError.textContent = "";
    }
    const previousRun = lastChallengeRun;
    const response = await fetch("/api/gpt/challenges/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      throw new Error("Failed to refresh GPT challenges");
    }
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

function startPolling(previousRun) {
  if (refreshTimer) {
    clearTimeout(refreshTimer);
  }
  let attempts = 0;
  const poll = async () => {
    attempts += 1;
    const data = await fetchRecommendations({ keepStatus: true });
    if (data) {
      if (data.status === "failed") {
        setStatus("error");
        setButtonLoading(gptRefresh, gptRefreshLabel, false);
        return;
      }
      if (data.run_time && data.run_time !== previousRun && data.status !== "running") {
        setStatus("idle");
        setButtonLoading(gptRefresh, gptRefreshLabel, false);
        return;
      }
    }
    if (attempts >= 12) {
      setStatus("error");
      setButtonLoading(gptRefresh, gptRefreshLabel, false);
      if (gptError) {
        gptError.textContent = "Refresh timed out. Try again in a moment.";
      }
      return;
    }
    refreshTimer = setTimeout(poll, 5000);
  };
  refreshTimer = setTimeout(poll, 3000);
}

function startChallengePolling(previousRun) {
  if (challengeTimer) {
    clearTimeout(challengeTimer);
  }
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
      if (gptChallengeError) {
        gptChallengeError.textContent = "Refresh timed out. Try again in a moment.";
      }
      return;
    }
    challengeTimer = setTimeout(poll, 5000);
  };
  challengeTimer = setTimeout(poll, 3000);
}

if (gptRefresh) {
  gptRefresh.addEventListener("click", refreshRecommendations);
}
if (gptChallengeRefresh) {
  gptChallengeRefresh.addEventListener("click", refreshChallenges);
}
document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (target.closest("#gpt-refresh")) {
    refreshRecommendations();
    return;
  }
  if (target.closest("#gpt-challenge-refresh")) {
    refreshChallenges();
  }
});
window.triggerGptRefresh = refreshRecommendations;
window.triggerGptChallengeRefresh = refreshChallenges;
fetchRecommendations();
fetchChallenges();
fetchFactorReviews();
fetchSummary();
