/* ═══════════════════════════════════════
   Tetra Dashboard — unified data manager
   ═══════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

// ── Formatting helpers ──
const fmt = {
  pct:  (v, d=1) => v == null ? '—' : (v * 100).toFixed(d) + '%',
  pctS: (v, d=1) => v == null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%',
  usd:  (v) => v == null ? '—' : '$' + Math.abs(v).toLocaleString('en-US', {maximumFractionDigits: 0}),
  usdS: (v) => v == null ? '—' : (v >= 0 ? '+$' : '-$') + Math.abs(v).toLocaleString('en-US', {maximumFractionDigits: 0}),
  num:  (v, d=1) => v == null ? '—' : Number(v).toFixed(d),
  price:(v) => v == null ? '—' : '$' + Number(v).toFixed(2),
};

// ── Data fetching ──
async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

// ── Render: Status Bar ──
function renderBar(risk, regime) {
  // Portfolio value
  if (risk) {
    const methods = risk.methods || {};
    const param = methods.parametric || {};
    const val = param.portfolio_value;
    if (val) {
      $('#port-value').textContent = fmt.usd(val);
    }
    $('#bar-asof').textContent = risk.as_of || '—';
  }

  // Regime pill
  if (regime) {
    const pill = $('#regime-pill');
    const state = regime.current_regime || 'unknown';
    pill.textContent = state;
    pill.className = 'regime-pill ' + state.toLowerCase();
  }
}

// ── Render: Regime Card ──
function renderRegime(data) {
  if (!data) return;

  const state = data.current_regime || '—';
  const el = $('#regime-state');
  el.textContent = state;
  el.className = 'regime-state ' + state.toLowerCase();

  // Probabilities
  const probs = data.current_probs || {};
  const probsEl = $('#regime-probs');
  probsEl.innerHTML = Object.entries(probs).map(([name, p]) =>
    `<span class="regime-prob"><span class="dot ${name.toLowerCase()}"></span>${name} ${fmt.pct(p, 0)}</span>`
  ).join('');

  // 5-day forecast
  const forecast = data.regime_forecast_5d || [];
  const forecastEl = $('#regime-forecast');
  const states = data.regime_states || [];
  const stateNames = states.map(s => s.label);
  if (forecast.length > 0 && stateNames.length > 0) {
    forecastEl.innerHTML = forecast.map((dayProbs, i) => {
      const fills = stateNames.map((name, si) => {
        const p = Array.isArray(dayProbs) ? (dayProbs[si] || 0) : (dayProbs[name] || 0);
        return `<div class="fill ${name.toLowerCase()}" style="height:${(p*100).toFixed(0)}%"></div>`;
      }).join('');
      return `<div class="forecast-bar">${fills}<span class="lbl">d${i+1}</span></div>`;
    }).join('');
  }
}

// ── Render: Risk Card ──
function renderRisk(data) {
  if (!data) return;
  const methods = data.methods || {};
  const p = methods.parametric || {};

  const volVal = p.total_vol_ann;
  const setKPI = (id, val, warnThresh, dangerThresh) => {
    const el = $(id);
    if (!el) return;
    el.textContent = val;
    el.classList.remove('warn', 'danger');
    if (dangerThresh != null && parseFloat(val) > dangerThresh) el.classList.add('danger');
    else if (warnThresh != null && parseFloat(val) > warnThresh) el.classList.add('warn');
  };

  setKPI('#risk-vol', fmt.pct(volVal), 0.20, 0.30);
  setKPI('#risk-var', fmt.usd(p.var_95_1d));
  setKPI('#risk-cvar', fmt.usd(p.cvar_95_1d));
  setKPI('#risk-mdd', fmt.pct(p.expected_max_drawdown), 0.15, 0.30);
  setKPI('#risk-effn', fmt.num(p.effective_n), null, null);
  setKPI('#risk-hhi', fmt.num(p.hhi, 2), null, null);

  // Breaches — from parametric risk_budget_breaches
  const breaches = p.risk_budget_breaches || {};
  const el = $('#risk-breaches');
  const tags = Object.keys(breaches).map(k =>
    `<span class="breach-tag">${k.replace(/_/g, ' ')}</span>`
  ).join('');
  el.innerHTML = tags;

  // Status bar breach count
  const n = Object.keys(breaches).length;
  const badge = $('#breach-badge');
  if (n > 0) {
    badge.style.display = '';
    $('#breach-count').textContent = n;
  } else {
    badge.style.display = 'none';
  }
}

// ── Render: Signals Card ──
function renderSignals(recs) {
  if (!recs || !recs.length) {
    $('#signal-counts').textContent = 'No signals';
    return;
  }

  let buy = 0, sell = 0, hold = 0;
  recs.forEach(r => {
    const a = (r.action || '').toLowerCase();
    if (a.includes('buy') || a.includes('long')) buy++;
    else if (a.includes('sell') || a.includes('short')) sell++;
    else hold++;
  });

  $('#signal-counts').innerHTML =
    `<span class="sig-buy">${buy} BUY</span>` +
    `<span class="sig-sell">${sell} SELL</span>` +
    `<span class="sig-hold">${hold} HOLD</span>`;

  // Top picks (first 4)
  const picks = recs.slice(0, 4);
  $('#signal-picks').innerHTML = picks.map(r => {
    const a = (r.action || '').toLowerCase();
    const cls = a.includes('buy') || a.includes('long') ? 'buy' : a.includes('sell') || a.includes('short') ? 'sell' : 'hold';
    return `<div class="pick-row">
      <span class="pick-sym">${r.symbol}</span>
      <span class="pick-action ${cls}">${(r.action||'').toUpperCase()}</span>
      <span class="pick-conf">${r.confidence ? Math.round(r.confidence * 100) + '%' : ''}</span>
    </div>`;
  }).join('');
}

// ── Render: Portfolio Card ──
function renderPortfolio(portfolio) {
  if (!portfolio || !portfolio.positions || !portfolio.positions.length) {
    $('#positions').innerHTML = '<div style="color:var(--dim)">No positions</div>';
    return;
  }

  const positions = portfolio.positions;
  const cash = portfolio.cash || 0;
  const total = portfolio.total_value || 0;
  const maxMV = Math.max(...positions.map(p => p.market_value));

  let html = positions.map(p => {
    const barW = maxMV > 0 ? (p.market_value / maxMV * 100).toFixed(0) : 0;
    const wPct = total > 0 ? p.market_value / total : 0;
    return `<div class="pos-row" data-tip="${p.shares} shares @ ${fmt.price(p.price)} = ${fmt.usd(p.market_value)}">
      <span class="pos-sym">${p.symbol}</span>
      <div class="pos-bar-wrap"><div class="pos-bar" style="width:${barW}%"></div></div>
      <span class="pos-weight">${fmt.pct(wPct, 1)}</span>
      <span class="pos-val">${fmt.usd(p.market_value)}</span>
    </div>`;
  }).join('');

  // Cash row
  if (cash > 0) {
    const cashW = total > 0 ? cash / total : 0;
    const cashBar = maxMV > 0 ? (cash / maxMV * 100).toFixed(0) : 0;
    html += `<div class="pos-row cash-row">
      <span class="pos-sym">CASH</span>
      <div class="pos-bar-wrap"><div class="pos-bar cash" style="width:${cashBar}%"></div></div>
      <span class="pos-weight">${fmt.pct(cashW, 1)}</span>
      <span class="pos-val">${fmt.usd(cash)}</span>
    </div>`;
  }

  $('#positions').innerHTML = html;

  // Update bar
  if (total > 0) {
    $('#port-value').textContent = fmt.usd(total);
  }
}

// ── Render: Recommendations Table (consensus only) ──
function renderRecs(data) {
  const body = $('#recs-body');
  const empty = $('#recs-empty');

  // Unified endpoint: debate + LLM consensus merged
  let recs = (data && data.consensus) || [];
  const debate = (data && data.debate) || {};

  if (!recs.length) {
    body.innerHTML = '';
    empty.style.display = '';
    empty.textContent = 'No recommendations available';
    $('#recs-meta').textContent = '';
    return [];
  }

  empty.style.display = 'none';
  const debateCount = recs.filter(r => (r.source || '').startsWith('debate')).length;
  const llmCount = recs.length - debateCount;
  let meta = `${recs.length} picks`;
  if (debateCount) meta += ` (${debateCount} debate`;
  if (llmCount && debateCount) meta += ` + ${llmCount} LLM consensus)`;
  else if (debateCount) meta += ')';
  if (debate.regime_consensus) meta += ` | Regime: ${debate.regime_consensus.slice(0, 60)}`;
  $('#recs-meta').textContent = meta;

  body.innerHTML = recs.map(r => {
    const a = (r.action || '').toLowerCase();
    const cls = a.includes('buy') || a.includes('long') ? 'buy' : a.includes('sell') || a.includes('short') ? 'sell' : 'hold';
    const provList = r.providers || [];
    const rr = r.reward_risk ? r.reward_risk.toFixed(1) + 'x' : '—';
    const confPct = r.confidence ? Math.round(r.confidence * 100) : 0;
    const confCls = confPct >= 70 ? 'conf-high' : confPct >= 50 ? 'conf-mid' : 'conf-low';
    // Source badge
    const src = r.source || 'llm_consensus';
    const srcLabel = src === 'debate_consensus' ? 'D' : src === 'debate_contrarian' ? 'C' : 'L';
    const srcTip = src === 'debate_consensus' ? 'Debate consensus — analysts agreed after 3 rounds'
                 : src === 'debate_contrarian' ? 'Debate contrarian — one analyst sees edge others miss'
                 : 'LLM consensus — independent provider agreement';
    const factorBadge = r.factor_aligned === true ? ' ✓F' : r.factor_aligned === false ? ' ✗F' : '';
    const factorTip = r.factor_aligned === true ? 'Factor signals AGREE with this trade'
                    : r.factor_aligned === false ? 'Factor signals DISAGREE — proceed with caution'
                    : '';
    // Provider/analyst detail
    const pConfs = r.provider_confidences || {};
    const provDetail = provList.map(p => {
      const pc = pConfs[p];
      return pc != null ? `${p}: ${Math.round(pc * 100)}%` : p;
    }).join(', ');
    const riskNote = r.risk ? ` | Risk: ${r.risk}` : '';
    return `<tr data-tip="${(r.thesis || '').replace(/"/g, '&quot;')}${riskNote}">
      <td class="sym"><span class="src-badge src-${srcLabel.toLowerCase()}" data-tip="${srcTip}">${srcLabel}</span> ${r.symbol || '—'}</td>
      <td class="${cls}">${(r.action||'—').toUpperCase()}</td>
      <td class="${confCls}" data-tip="Confidence: ${confPct}% | ${provDetail}${factorTip ? ' | ' + factorTip : ''}">${confPct}%${factorBadge}</td>
      <td>${fmt.price(r.entry)}</td>
      <td>${fmt.price(r.target)}</td>
      <td>${fmt.price(r.stop)}</td>
      <td class="${r.expected_return >= 0 ? 'buy' : 'sell'}">${r.expected_return ? fmt.pctS(r.expected_return) : '—'}</td>
      <td>${rr}</td>
      <td style="color:var(--dim)" data-tip="${provDetail}">${r.support || 0}/${provList.join(', ')}</td>
    </tr>`;
  }).join('');

  return recs;
}

// ── Render: Scenario Chart ──
function renderScenarios(data) {
  const container = $('#scenario-chart');
  if (!data || !data.scenarios || !data.scenarios.length) {
    container.innerHTML = '<div class="empty-state">No scenario data</div>';
    return;
  }

  const scenarios = data.scenarios;
  $('#scenario-count').textContent = `${scenarios.length} scenarios`;

  const maxAbs = Math.max(...scenarios.map(s => Math.abs(s.pnl_pct || 0)), 0.01);

  container.innerHTML = scenarios.map((s, i) => {
    const pct = s.pnl_pct || 0;
    const isNeg = pct < 0;
    const barWidth = (Math.abs(pct) / maxAbs * 48).toFixed(1); // max 48% of container width
    const barLeft = isNeg ? (50 - parseFloat(barWidth)).toFixed(1) : '50';

    return `<div class="sc-row" data-idx="${i}">
      <span class="sc-name" title="${s.description || ''}">${s.name}</span>
      <div class="sc-bar-wrap">
        <div class="sc-bar-center"></div>
        <div class="sc-bar ${isNeg ? 'negative' : 'positive'}" style="left:${barLeft}%;width:${barWidth}%"></div>
      </div>
      <span class="sc-pnl ${isNeg ? 'negative' : 'positive'}">${fmt.pctS(pct)}</span>
    </div>`;
  }).join('') + '<div class="sc-detail" id="sc-detail"></div>';

  // Click to show detail
  container.querySelectorAll('.sc-row').forEach(row => {
    row.style.cursor = 'pointer';
    row.addEventListener('click', () => {
      const idx = parseInt(row.dataset.idx);
      const s = scenarios[idx];
      const detail = $('#sc-detail');
      if (detail.dataset.active === String(idx)) {
        detail.classList.remove('active');
        detail.dataset.active = '';
        return;
      }
      detail.dataset.active = String(idx);

      const positions = s.position_pnls || {};
      const posHtml = Object.entries(positions)
        .filter(([, v]) => Math.abs(v) > 0)
        .sort((a, b) => a[1] - b[1])
        .map(([sym, pnl]) => `<strong>${sym}</strong> ${fmt.usdS(pnl)}`)
        .join(' · ');

      detail.innerHTML = `
        <div><strong>${s.name}</strong> — ${s.description || ''}</div>
        <div style="margin-top:4px">Portfolio: <strong>${fmt.usdS(s.pnl)}</strong> (${fmt.pctS(s.pnl_pct)})</div>
        ${posHtml ? `<div style="margin-top:4px">Positions: ${posHtml}</div>` : ''}
        ${s.worst_position ? `<div style="margin-top:4px">Worst: <strong>${s.worst_position}</strong> ${fmt.usdS(s.worst_pnl)}</div>` : ''}
      `;
      detail.classList.add('active');
    });
  });
}

// ── Render: Correlation Heatmap ──
function renderCorrelation(data) {
  const canvas = $('#corr-canvas');
  if (!data || !data.correlation_matrix || !data.symbols) {
    canvas.style.display = 'none';
    return;
  }

  const symbols = data.symbols;
  const corr = data.correlation_matrix;
  const n = symbols.length;

  // Filter to portfolio + major symbols for readability
  const keep = new Set(['SPY', 'QQQ', 'IWM', 'NVDA', 'META', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA',
                        'XLK', 'XLE', 'XLF', 'TLT', 'GLD', 'UVXY', 'HYG', 'IBIT']);
  const indices = [];
  symbols.forEach((s, i) => { if (keep.has(s)) indices.push(i); });

  const m = indices.length;
  if (m === 0) return;

  const cellSize = Math.min(28, Math.floor(600 / m));
  const labelW = 44;
  const W = labelW + m * cellSize;
  const H = labelW + m * cellSize;

  canvas.width = W * 2;
  canvas.height = H * 2;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, W, H);

  // Draw cells
  for (let ri = 0; ri < m; ri++) {
    for (let ci = 0; ci < m; ci++) {
      const val = corr[indices[ri]][indices[ci]];
      ctx.fillStyle = corrColor(val);
      ctx.fillRect(labelW + ci * cellSize, labelW + ri * cellSize, cellSize - 1, cellSize - 1);

      // Value text for larger cells
      if (cellSize >= 22 && ri !== ci) {
        ctx.fillStyle = Math.abs(val) > 0.5 ? '#fff' : 'rgba(255,255,255,0.5)';
        ctx.font = `${Math.min(9, cellSize * 0.35)}px JetBrains Mono`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(2), labelW + ci * cellSize + cellSize / 2, labelW + ri * cellSize + cellSize / 2);
      }
    }
  }

  // Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < m; i++) {
    ctx.fillText(symbols[indices[i]], labelW - 4, labelW + i * cellSize + cellSize / 2);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i < m; i++) {
    ctx.save();
    ctx.translate(labelW + i * cellSize + cellSize / 2, labelW - 4);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = 'right';
    ctx.fillText(symbols[indices[i]], 0, 0);
    ctx.restore();
  }
}

function corrColor(v) {
  if (v >= 0) {
    const t = Math.min(v, 1);
    return `rgba(16,185,129,${0.15 + t * 0.75})`;
  } else {
    const t = Math.min(-v, 1);
    return `rgba(239,68,68,${0.15 + t * 0.75})`;
  }
}

// ── Render: Volatility Chart ──
function renderVolatility(data) {
  const canvas = $('#vol-canvas');
  if (!data || !data.symbols || !data.vols_ann || !data.vols_ann.length) {
    canvas.style.display = 'none';
    return;
  }

  const symbols = data.symbols;
  const vols = data.vols_ann;
  const pairs = symbols.map((s, i) => [s, vols[i]]).sort((a, b) => b[1] - a[1]);

  // Top 20
  const top = pairs.slice(0, 20);
  const n = top.length;
  const barH = 16;
  const gap = 4;
  const labelW = 48;
  const rightPad = 50;
  const W = 600;
  const H = labelW + n * (barH + gap);

  canvas.width = W * 2;
  canvas.height = H * 2;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, W, H);

  const maxVol = Math.max(...top.map(p => p[1]));
  const barMaxW = W - labelW - rightPad;

  top.forEach(([sym, vol], i) => {
    const y = 8 + i * (barH + gap);
    const w = (vol / maxVol) * barMaxW;

    const color = vol > 0.5 ? 'rgba(239,68,68,0.65)' : vol > 0.3 ? 'rgba(245,158,11,0.55)' : 'rgba(59,130,246,0.50)';
    ctx.fillStyle = color;
    ctx.fillRect(labelW, y, w, barH);

    // Label
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(sym, labelW - 6, y + barH / 2);

    // Value
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText((vol * 100).toFixed(1) + '%', labelW + w + 6, y + barH / 2);
  });
}

// ── Render: Factor Signals (deep dive) ──
function renderFactors(data) {
  const body = $('#factor-body');
  if (!data) { body.innerHTML = ''; return; }

  const all = [...(data.top_longs || []), ...(data.top_shorts || [])];
  all.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));

  body.innerHTML = all.map(f => {
    const a = (f.action || '').toLowerCase();
    const cls = a.includes('buy') ? 'buy' : a.includes('sell') ? 'sell' : 'hold';
    return `<tr>
      <td class="sym">${f.symbol}</td>
      <td>${f.score != null ? f.score.toFixed(3) : '—'}</td>
      <td class="${cls}">${(f.action || '—').toUpperCase()}</td>
      <td>${f.coverage || '—'}</td>
    </tr>`;
  }).join('');
}

// ── Render: IC Leaderboard (deep dive) ──
function renderInference(data) {
  const body = $('#ic-body');
  if (!data || !data.leaderboard) { body.innerHTML = ''; return; }

  body.innerHTML = data.leaderboard.map(r => `<tr>
    <td>${r.factor}</td>
    <td>${r.horizon || '—'}</td>
    <td>${r.avg_ic != null ? r.avg_ic.toFixed(4) : '—'}</td>
    <td>${r.median_ic != null ? r.median_ic.toFixed(4) : '—'}</td>
    <td>${r.hit_rate != null ? fmt.pct(r.hit_rate) : '—'}</td>
    <td>${r.days || '—'}</td>
  </tr>`).join('');
}

// ── Render: Risk Contributors (deep dive) ──
function renderContributors(risk) {
  const container = $('#contrib-chart');
  if (!risk) return;
  const p = (risk.methods || {}).parametric || {};
  const comp = p.component_risk || {};

  const entries = Object.entries(comp)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 15);

  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.001);

  container.innerHTML = entries.map(([sym, cr]) => {
    const barW = (Math.abs(cr) / maxAbs * 100).toFixed(0);
    const isNeg = cr < 0;
    return `<div class="contrib-row">
      <span class="contrib-sym">${sym}</span>
      <div class="contrib-bar-wrap"><div class="contrib-bar ${isNeg ? 'negative' : ''}" style="width:${barW}%"></div></div>
      <span class="contrib-pct">${fmt.pct(cr, 2)}</span>
    </div>`;
  }).join('');
}

// ── Main data loader ──
async function loadAll() {
  const [regime, risk, scenarios, covariance, recs, factors, inference, portfolio] = await Promise.all([
    fetchJSON('/api/simulation/regime'),
    fetchJSON('/api/simulation/risk'),
    fetchJSON('/api/simulation/scenarios'),
    fetchJSON('/api/simulation/covariance?method=ledoit_wolf_identity'),
    fetchJSON('/api/gpt/consensus'),
    fetchJSON('/api/factors/alpha'),
    fetchJSON('/api/inference/signal-leaderboard'),
    fetchJSON('/api/portfolio'),
  ]);

  renderBar(risk, regime);
  renderRegime(regime);
  renderRisk(risk);
  renderPortfolio(portfolio);

  const recList = renderRecs(recs);
  renderSignals(recList);

  renderScenarios(scenarios);
  renderCorrelation(covariance);
  renderVolatility(covariance);
  renderFactors(factors);
  renderInference(inference);
  renderContributors(risk);
}

// ── Init ──
loadAll();

// Auto-refresh every 60s
setInterval(loadAll, 60000);

// Manual refresh
$('#btn-refresh').addEventListener('click', () => {
  $('#btn-refresh').style.opacity = '0.4';
  loadAll().then(() => { $('#btn-refresh').style.opacity = '1'; });
});

// ── Tooltip engine ──
(function initTooltip() {
  const tip = document.createElement('div');
  tip.className = 'tip';
  document.body.appendChild(tip);

  let active = null;
  let showTimer = null;

  function pos(e) {
    const pad = 12;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - pad) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - pad) y = e.clientY - r.height - pad;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }

  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest('[data-tip]');
    if (!el || el === active) return;
    active = el;
    clearTimeout(showTimer);
    showTimer = setTimeout(() => {
      tip.textContent = el.dataset.tip;
      tip.classList.add('visible');
      pos(e);
    }, 350);
  });

  document.addEventListener('mousemove', (e) => {
    if (tip.classList.contains('visible')) pos(e);
  });

  document.addEventListener('mouseout', (e) => {
    const el = e.target.closest('[data-tip]');
    if (el === active) {
      clearTimeout(showTimer);
      active = null;
      tip.classList.remove('visible');
    }
  });
})();
